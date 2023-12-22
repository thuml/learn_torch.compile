
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


cpp_fused_add_arange_embedding_mul_native_layer_norm_0 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0)];
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                        auto tmp1 = decltype(tmp0)(tmp0 + 8008);
                        auto tmp2 = tmp0 < 0;
                        auto tmp3 = tmp2 ? tmp1 : tmp0;
                        TORCH_CHECK((0 <= tmp3) & (tmp3 < 8008L), "index out of bounds: 0 <= tmp3 < 8008L")
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2560L*tmp3)));
                        auto tmp5 = static_cast<float>(1.0);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp9 = tmp7 + tmp8;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp9);
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x0)];
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp10 = out_ptr0[static_cast<long>(x0)];
                    auto tmp13 = out_ptr1[static_cast<long>(x0)];
                    auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp1 = decltype(tmp0)(tmp0 + 8008);
                    auto tmp2 = tmp0 < 0;
                    auto tmp3 = tmp2 ? tmp1 : tmp0;
                    TORCH_CHECK((0 <= tmp3) & (tmp3 < 8008L), "index out of bounds: 0 <= tmp3 < 8008L")
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2560L*tmp3)));
                    auto tmp5 = static_cast<float>(1.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp9 = tmp7 + tmp8;
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 - tmp11;
                    auto tmp14 = static_cast<float>(2560.0);
                    auto tmp15 = tmp13 / tmp14;
                    auto tmp16 = static_cast<float>(1e-05);
                    auto tmp17 = decltype(tmp15)(tmp15 + tmp16);
                    auto tmp18 = 1 / std::sqrt(tmp17);
                    auto tmp19 = at::vec::Vectorized<float>(tmp18);
                    auto tmp20 = tmp12 * tmp19;
                    auto tmp22 = tmp20 * tmp21;
                    auto tmp24 = tmp22 + tmp23;
                    tmp24.store(out_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_1 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        auto tmp1 = static_cast<float>(0.11180339887498948);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr0 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        tmp0.store(out_ptr2 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_2 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(327680L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_arange_embedding_mul_native_layer_norm_3 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const long* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const long* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    Welford<float> tmp_acc1 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc1_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0)];
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2560L*x0)));
                        auto tmp12 = in_ptr4[static_cast<long>(x0)];
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (2560L*x0)));
                        auto tmp1 = decltype(tmp0)(tmp0 + 8008);
                        auto tmp2 = tmp0 < 0;
                        auto tmp3 = tmp2 ? tmp1 : tmp0;
                        TORCH_CHECK((0 <= tmp3) & (tmp3 < 8008L), "index out of bounds: 0 <= tmp3 < 8008L")
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2560L*tmp3)));
                        auto tmp5 = static_cast<float>(1.0);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp9 = tmp7 + tmp8;
                        auto tmp11 = tmp9 + tmp10;
                        auto tmp13 = decltype(tmp12)(tmp12 + 8008);
                        auto tmp14 = tmp12 < 0;
                        auto tmp15 = tmp14 ? tmp13 : tmp12;
                        TORCH_CHECK((0 <= tmp15) & (tmp15 < 8008L), "index out of bounds: 0 <= tmp15 < 8008L")
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2560L*tmp15)));
                        auto tmp17 = tmp16 * tmp6;
                        auto tmp19 = tmp17 + tmp18;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp11);
                        tmp_acc1_vec = welford_combine(tmp_acc1_vec, tmp19);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                    tmp_acc1 = welford_combine(tmp_acc1, welford_vec_reduce_all(tmp_acc1_vec));
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc1.mean);
                    out_ptr3[static_cast<long>(x0)] = static_cast<float>(tmp_acc1.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x0)];
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp12 = out_ptr0[static_cast<long>(x0)];
                    auto tmp15 = out_ptr1[static_cast<long>(x0)];
                    auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp1 = decltype(tmp0)(tmp0 + 8008);
                    auto tmp2 = tmp0 < 0;
                    auto tmp3 = tmp2 ? tmp1 : tmp0;
                    TORCH_CHECK((0 <= tmp3) & (tmp3 < 8008L), "index out of bounds: 0 <= tmp3 < 8008L")
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2560L*tmp3)));
                    auto tmp5 = static_cast<float>(1.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp9 = tmp7 + tmp8;
                    auto tmp11 = tmp9 + tmp10;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 - tmp13;
                    auto tmp16 = static_cast<float>(2560.0);
                    auto tmp17 = tmp15 / tmp16;
                    auto tmp18 = static_cast<float>(1e-05);
                    auto tmp19 = decltype(tmp17)(tmp17 + tmp18);
                    auto tmp20 = 1 / std::sqrt(tmp19);
                    auto tmp21 = at::vec::Vectorized<float>(tmp20);
                    auto tmp22 = tmp14 * tmp21;
                    auto tmp24 = tmp22 * tmp23;
                    auto tmp26 = tmp24 + tmp25;
                    tmp26.store(out_ptr4 + static_cast<long>(x1 + (2560L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_4 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1310720L); x0+=static_cast<long>(8L))
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


cpp_fused_add_arange_embedding_mul_native_layer_norm_5 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const long* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const long* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       const float* in_ptr8,
                       const float* in_ptr9,
                       const float* in_ptr10,
                       const float* in_ptr11,
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0)];
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2560L*x0)));
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                        auto tmp14 = in_ptr4[static_cast<long>(x0)];
                        auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (2560L*x0)));
                        auto tmp22 = in_ptr6[static_cast<long>(x0)];
                        auto tmp25 = in_ptr7[static_cast<long>(x0)];
                        auto tmp33 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                        auto tmp35 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1));
                        auto tmp1 = decltype(tmp0)(tmp0 + 8008);
                        auto tmp2 = tmp0 < 0;
                        auto tmp3 = tmp2 ? tmp1 : tmp0;
                        TORCH_CHECK((0 <= tmp3) & (tmp3 < 8008L), "index out of bounds: 0 <= tmp3 < 8008L")
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2560L*tmp3)));
                        auto tmp5 = static_cast<float>(1.0);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp9 = tmp7 + tmp8;
                        auto tmp11 = tmp9 + tmp10;
                        auto tmp13 = tmp11 + tmp12;
                        auto tmp15 = decltype(tmp14)(tmp14 + 8008);
                        auto tmp16 = tmp14 < 0;
                        auto tmp17 = tmp16 ? tmp15 : tmp14;
                        TORCH_CHECK((0 <= tmp17) & (tmp17 < 8008L), "index out of bounds: 0 <= tmp17 < 8008L")
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2560L*tmp17)));
                        auto tmp19 = tmp18 * tmp6;
                        auto tmp21 = tmp19 + tmp20;
                        auto tmp23 = at::vec::Vectorized<float>(tmp22);
                        auto tmp24 = tmp21 - tmp23;
                        auto tmp26 = static_cast<float>(2560.0);
                        auto tmp27 = tmp25 / tmp26;
                        auto tmp28 = static_cast<float>(1e-05);
                        auto tmp29 = decltype(tmp27)(tmp27 + tmp28);
                        auto tmp30 = 1 / std::sqrt(tmp29);
                        auto tmp31 = at::vec::Vectorized<float>(tmp30);
                        auto tmp32 = tmp24 * tmp31;
                        auto tmp34 = tmp32 * tmp33;
                        auto tmp36 = tmp34 + tmp35;
                        tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                        tmp36.store(out_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp13);
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = out_ptr2[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(2560.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (2560L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_6 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        auto tmp1 = static_cast<float>(0.11180339887498948);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr0 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        tmp0.store(out_ptr2 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_7 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(327680L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(in_out_ptr0 + static_cast<long>(x0));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2560L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(2560.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (2560L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1310720L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_10 = async_compile.cpp('''
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
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2560L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2560L*x0)));
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
    }
}
''')


cpp_fused_clone_11 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        auto tmp1 = static_cast<float>(0.11180339887498948);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr0 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_12 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))];
                            auto tmp1 = c10::convert<long>(x2);
                            auto tmp2 = c10::convert<long>(1L + x1);
                            auto tmp3 = tmp1 < tmp2;
                            auto tmp4 = static_cast<float>(0.0);
                            auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp6 = tmp3 ? tmp4 : tmp5;
                            auto tmp7 = decltype(tmp0)(tmp0 + tmp6);
                            tmp_acc0 = max_propagate_nan(tmp_acc0, tmp7);
                        }
                        out_ptr0[static_cast<long>(x1 + (128L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))];
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp1 = c10::convert<long>(x2);
                        auto tmp2 = c10::convert<long>(1L + x1);
                        auto tmp3 = tmp1 < tmp2;
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp6 = tmp3 ? tmp4 : tmp5;
                        auto tmp7 = decltype(tmp0)(tmp0 + tmp6);
                        auto tmp9 = decltype(tmp7)(tmp7 - tmp8);
                        auto tmp10 = std::exp(tmp9);
                        in_out_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))] = tmp10;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
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


cpp_fused__softmax_clone_13 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_14 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (80L*x0) + (10240L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (80L*x1) + (2560L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_arange_embedding_mul_native_layer_norm_15 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const long* in_ptr0,
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0)];
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2560L*x0)));
                        auto tmp1 = decltype(tmp0)(tmp0 + 8008);
                        auto tmp2 = tmp0 < 0;
                        auto tmp3 = tmp2 ? tmp1 : tmp0;
                        TORCH_CHECK((0 <= tmp3) & (tmp3 < 8008L), "index out of bounds: 0 <= tmp3 < 8008L")
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2560L*tmp3)));
                        auto tmp5 = static_cast<float>(1.0);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp9 = tmp7 + tmp8;
                        auto tmp11 = tmp9 + tmp10;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp11);
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x0)];
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp12 = out_ptr0[static_cast<long>(x0)];
                    auto tmp15 = out_ptr1[static_cast<long>(x0)];
                    auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = decltype(tmp0)(tmp0 + 8008);
                    auto tmp2 = tmp0 < 0;
                    auto tmp3 = tmp2 ? tmp1 : tmp0;
                    TORCH_CHECK((0 <= tmp3) & (tmp3 < 8008L), "index out of bounds: 0 <= tmp3 < 8008L")
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2560L*tmp3)));
                    auto tmp5 = static_cast<float>(1.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp9 = tmp7 + tmp8;
                    auto tmp11 = tmp9 + tmp10;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 - tmp13;
                    auto tmp16 = static_cast<float>(2560.0);
                    auto tmp17 = tmp15 / tmp16;
                    auto tmp18 = static_cast<float>(1e-05);
                    auto tmp19 = decltype(tmp17)(tmp17 + tmp18);
                    auto tmp20 = 1 / std::sqrt(tmp19);
                    auto tmp21 = at::vec::Vectorized<float>(tmp20);
                    auto tmp22 = tmp14 * tmp21;
                    auto tmp24 = tmp22 * tmp23;
                    auto tmp26 = tmp24 + tmp25;
                    tmp26.store(out_ptr2 + static_cast<long>(x1 + (2560L*x0)));
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
                       const float* in_ptr6,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp5 = in_ptr3[static_cast<long>(x0)];
                    auto tmp8 = in_ptr4[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(2560.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-05);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(out_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_17 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        auto tmp1 = static_cast<float>(0.11180339887498948);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr0 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        tmp0.store(out_ptr2 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_18 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(327680L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_arange_embedding_mul_native_layer_norm_19 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const long* in_ptr0,
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0)];
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2560L*x0)));
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                        auto tmp1 = decltype(tmp0)(tmp0 + 8008);
                        auto tmp2 = tmp0 < 0;
                        auto tmp3 = tmp2 ? tmp1 : tmp0;
                        TORCH_CHECK((0 <= tmp3) & (tmp3 < 8008L), "index out of bounds: 0 <= tmp3 < 8008L")
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2560L*tmp3)));
                        auto tmp5 = static_cast<float>(1.0);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp9 = tmp7 + tmp8;
                        auto tmp11 = tmp9 + tmp10;
                        auto tmp13 = tmp11 + tmp12;
                        tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp13);
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(2560.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_20 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1310720L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2560L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(2560.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_22 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        auto tmp1 = static_cast<float>(0.11180339887498948);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr0 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_23 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))];
                            auto tmp1 = c10::convert<long>(x2);
                            auto tmp2 = c10::convert<long>(1L + x1);
                            auto tmp3 = tmp1 < tmp2;
                            auto tmp4 = static_cast<float>(0.0);
                            auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp6 = tmp3 ? tmp4 : tmp5;
                            auto tmp7 = decltype(tmp0)(tmp0 + tmp6);
                            tmp_acc0 = max_propagate_nan(tmp_acc0, tmp7);
                        }
                        out_ptr0[static_cast<long>(x1 + (128L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))];
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp1 = c10::convert<long>(x2);
                        auto tmp2 = c10::convert<long>(1L + x1);
                        auto tmp3 = tmp1 < tmp2;
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp6 = tmp3 ? tmp4 : tmp5;
                        auto tmp7 = decltype(tmp0)(tmp0 + tmp6);
                        auto tmp9 = decltype(tmp7)(tmp7 - tmp8);
                        auto tmp10 = std::exp(tmp9);
                        in_out_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))] = tmp10;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
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


cpp_fused__softmax_clone_24 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_25 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (80L*x0) + (10240L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (80L*x1) + (2560L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_26 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2560L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2560L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(2560.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-05);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (2560L*x0)));
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
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        auto tmp1 = static_cast<float>(0.11180339887498948);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr0 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        tmp0.store(out_ptr2 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_28 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(327680L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_29 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2560L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2560L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(2560.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-05);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp21.store(out_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_30 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1310720L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_31 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(327680L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2560L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(2560.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_32 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        auto tmp1 = static_cast<float>(0.11180339887498948);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr0 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_33 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))];
                            auto tmp1 = c10::convert<long>(x2);
                            auto tmp2 = c10::convert<long>(1L + x1);
                            auto tmp3 = tmp1 < tmp2;
                            auto tmp4 = static_cast<float>(0.0);
                            auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp6 = tmp3 ? tmp4 : tmp5;
                            auto tmp7 = decltype(tmp0)(tmp0 + tmp6);
                            tmp_acc0 = max_propagate_nan(tmp_acc0, tmp7);
                        }
                        out_ptr0[static_cast<long>(x1 + (128L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))];
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp1 = c10::convert<long>(x2);
                        auto tmp2 = c10::convert<long>(1L + x1);
                        auto tmp3 = tmp1 < tmp2;
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp6 = tmp3 ? tmp4 : tmp5;
                        auto tmp7 = decltype(tmp0)(tmp0 + tmp6);
                        auto tmp9 = decltype(tmp7)(tmp7 - tmp8);
                        auto tmp10 = std::exp(tmp9);
                        in_out_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))] = tmp10;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
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


cpp_fused__softmax_clone_34 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
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
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (80L*x0) + (10240L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (80L*x1) + (2560L*x0)));
                    }
                }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2560L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(2560.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_37 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        auto tmp1 = static_cast<float>(0.11180339887498948);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr0 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        tmp0.store(out_ptr2 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_38 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(327680L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_39 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2560L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2560L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(2560.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-05);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_40 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1310720L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_41 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2560L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2560L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(2560.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-05);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp21.store(out_ptr2 + static_cast<long>(x1 + (2560L*x0)));
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
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        auto tmp1 = static_cast<float>(0.11180339887498948);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr0 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_43 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))];
                            auto tmp1 = c10::convert<long>(x2);
                            auto tmp2 = c10::convert<long>(1L + x1);
                            auto tmp3 = tmp1 < tmp2;
                            auto tmp4 = static_cast<float>(0.0);
                            auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp6 = tmp3 ? tmp4 : tmp5;
                            auto tmp7 = decltype(tmp0)(tmp0 + tmp6);
                            tmp_acc0 = max_propagate_nan(tmp_acc0, tmp7);
                        }
                        out_ptr0[static_cast<long>(x1 + (128L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))];
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp1 = c10::convert<long>(x2);
                        auto tmp2 = c10::convert<long>(1L + x1);
                        auto tmp3 = tmp1 < tmp2;
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp6 = tmp3 ? tmp4 : tmp5;
                        auto tmp7 = decltype(tmp0)(tmp0 + tmp6);
                        auto tmp9 = decltype(tmp7)(tmp7 - tmp8);
                        auto tmp10 = std::exp(tmp9);
                        in_out_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))] = tmp10;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
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


cpp_fused__softmax_clone_44 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (80L*x0) + (10240L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (80L*x1) + (2560L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_46 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(327680L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2560L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(2560.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_47 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        auto tmp1 = static_cast<float>(0.11180339887498948);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr0 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        tmp0.store(out_ptr2 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_48 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(327680L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_49 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2560L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(2560.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_50 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1310720L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_51 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2560L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2560L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(2560.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-05);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_52 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        auto tmp1 = static_cast<float>(0.11180339887498948);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr0 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_53 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))];
                            auto tmp1 = c10::convert<long>(x2);
                            auto tmp2 = c10::convert<long>(1L + x1);
                            auto tmp3 = tmp1 < tmp2;
                            auto tmp4 = static_cast<float>(0.0);
                            auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp6 = tmp3 ? tmp4 : tmp5;
                            auto tmp7 = decltype(tmp0)(tmp0 + tmp6);
                            tmp_acc0 = max_propagate_nan(tmp_acc0, tmp7);
                        }
                        out_ptr0[static_cast<long>(x1 + (128L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))];
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp1 = c10::convert<long>(x2);
                        auto tmp2 = c10::convert<long>(1L + x1);
                        auto tmp3 = tmp1 < tmp2;
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp6 = tmp3 ? tmp4 : tmp5;
                        auto tmp7 = decltype(tmp0)(tmp0 + tmp6);
                        auto tmp9 = decltype(tmp7)(tmp7 - tmp8);
                        auto tmp10 = std::exp(tmp9);
                        in_out_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))] = tmp10;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
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


cpp_fused__softmax_clone_54 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_55 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (80L*x0) + (10240L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (80L*x1) + (2560L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_56 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2560L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2560L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(2560.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-05);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp21.store(out_ptr2 + static_cast<long>(x1 + (2560L*x0)));
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
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        auto tmp1 = static_cast<float>(0.11180339887498948);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr0 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        tmp0.store(out_ptr2 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_58 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(327680L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_59 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(327680L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2560L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(2560.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_60 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1310720L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_61 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2560L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(2560.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_62 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        auto tmp1 = static_cast<float>(0.11180339887498948);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr0 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_63 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))];
                            auto tmp1 = c10::convert<long>(x2);
                            auto tmp2 = c10::convert<long>(1L + x1);
                            auto tmp3 = tmp1 < tmp2;
                            auto tmp4 = static_cast<float>(0.0);
                            auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp6 = tmp3 ? tmp4 : tmp5;
                            auto tmp7 = decltype(tmp0)(tmp0 + tmp6);
                            tmp_acc0 = max_propagate_nan(tmp_acc0, tmp7);
                        }
                        out_ptr0[static_cast<long>(x1 + (128L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))];
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp1 = c10::convert<long>(x2);
                        auto tmp2 = c10::convert<long>(1L + x1);
                        auto tmp3 = tmp1 < tmp2;
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp6 = tmp3 ? tmp4 : tmp5;
                        auto tmp7 = decltype(tmp0)(tmp0 + tmp6);
                        auto tmp9 = decltype(tmp7)(tmp7 - tmp8);
                        auto tmp10 = std::exp(tmp9);
                        in_out_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))] = tmp10;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
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


cpp_fused__softmax_clone_64 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_65 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (80L*x0) + (10240L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (80L*x1) + (2560L*x0)));
                    }
                }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2560L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2560L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(2560.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-05);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_67 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        auto tmp1 = static_cast<float>(0.11180339887498948);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr0 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        tmp0.store(out_ptr2 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_68 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(327680L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_69 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2560L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2560L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(2560.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-05);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp21.store(out_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_70 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1310720L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_71 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(327680L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2560L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(2560.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_72 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        auto tmp1 = static_cast<float>(0.11180339887498948);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr0 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_73 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))];
                            auto tmp1 = c10::convert<long>(x2);
                            auto tmp2 = c10::convert<long>(1L + x1);
                            auto tmp3 = tmp1 < tmp2;
                            auto tmp4 = static_cast<float>(0.0);
                            auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp6 = tmp3 ? tmp4 : tmp5;
                            auto tmp7 = decltype(tmp0)(tmp0 + tmp6);
                            tmp_acc0 = max_propagate_nan(tmp_acc0, tmp7);
                        }
                        out_ptr0[static_cast<long>(x1 + (128L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))];
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp1 = c10::convert<long>(x2);
                        auto tmp2 = c10::convert<long>(1L + x1);
                        auto tmp3 = tmp1 < tmp2;
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp6 = tmp3 ? tmp4 : tmp5;
                        auto tmp7 = decltype(tmp0)(tmp0 + tmp6);
                        auto tmp9 = decltype(tmp7)(tmp7 - tmp8);
                        auto tmp10 = std::exp(tmp9);
                        in_out_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))] = tmp10;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
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


cpp_fused__softmax_clone_74 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (80L*x0) + (10240L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (80L*x1) + (2560L*x0)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2560L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(2560.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_77 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        auto tmp1 = static_cast<float>(0.11180339887498948);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr0 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        tmp0.store(out_ptr2 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_78 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(327680L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_79 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2560L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2560L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(2560.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-05);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_80 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1310720L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_81 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2560L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2560L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(2560.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-05);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp21.store(out_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_82 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        auto tmp1 = static_cast<float>(0.11180339887498948);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr0 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_83 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))];
                            auto tmp1 = c10::convert<long>(x2);
                            auto tmp2 = c10::convert<long>(1L + x1);
                            auto tmp3 = tmp1 < tmp2;
                            auto tmp4 = static_cast<float>(0.0);
                            auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp6 = tmp3 ? tmp4 : tmp5;
                            auto tmp7 = decltype(tmp0)(tmp0 + tmp6);
                            tmp_acc0 = max_propagate_nan(tmp_acc0, tmp7);
                        }
                        out_ptr0[static_cast<long>(x1 + (128L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))];
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp1 = c10::convert<long>(x2);
                        auto tmp2 = c10::convert<long>(1L + x1);
                        auto tmp3 = tmp1 < tmp2;
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp6 = tmp3 ? tmp4 : tmp5;
                        auto tmp7 = decltype(tmp0)(tmp0 + tmp6);
                        auto tmp9 = decltype(tmp7)(tmp7 - tmp8);
                        auto tmp10 = std::exp(tmp9);
                        in_out_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))] = tmp10;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
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


cpp_fused__softmax_clone_84 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_85 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (80L*x0) + (10240L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (80L*x1) + (2560L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_86 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(327680L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2560L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(2560.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_87 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        auto tmp1 = static_cast<float>(0.11180339887498948);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr0 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        tmp0.store(out_ptr2 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_88 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(327680L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_89 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2560L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(2560.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_90 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1310720L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_91 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2560L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2560L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(2560.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-05);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_92 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        auto tmp1 = static_cast<float>(0.11180339887498948);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr0 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_93 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))];
                            auto tmp1 = c10::convert<long>(x2);
                            auto tmp2 = c10::convert<long>(1L + x1);
                            auto tmp3 = tmp1 < tmp2;
                            auto tmp4 = static_cast<float>(0.0);
                            auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp6 = tmp3 ? tmp4 : tmp5;
                            auto tmp7 = decltype(tmp0)(tmp0 + tmp6);
                            tmp_acc0 = max_propagate_nan(tmp_acc0, tmp7);
                        }
                        out_ptr0[static_cast<long>(x1 + (128L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))];
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp1 = c10::convert<long>(x2);
                        auto tmp2 = c10::convert<long>(1L + x1);
                        auto tmp3 = tmp1 < tmp2;
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp6 = tmp3 ? tmp4 : tmp5;
                        auto tmp7 = decltype(tmp0)(tmp0 + tmp6);
                        auto tmp9 = decltype(tmp7)(tmp7 - tmp8);
                        auto tmp10 = std::exp(tmp9);
                        in_out_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))] = tmp10;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
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


cpp_fused__softmax_clone_94 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_95 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (80L*x0) + (10240L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (80L*x1) + (2560L*x0)));
                    }
                }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2560L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2560L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(2560.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-05);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp21.store(out_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_97 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        auto tmp1 = static_cast<float>(0.11180339887498948);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr0 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        tmp0.store(out_ptr2 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_98 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(327680L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_99 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(327680L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2560L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(2560.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_100 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1310720L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_101 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2560L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(2560.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_102 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        auto tmp1 = static_cast<float>(0.11180339887498948);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr0 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_103 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))];
                            auto tmp1 = c10::convert<long>(x2);
                            auto tmp2 = c10::convert<long>(1L + x1);
                            auto tmp3 = tmp1 < tmp2;
                            auto tmp4 = static_cast<float>(0.0);
                            auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp6 = tmp3 ? tmp4 : tmp5;
                            auto tmp7 = decltype(tmp0)(tmp0 + tmp6);
                            tmp_acc0 = max_propagate_nan(tmp_acc0, tmp7);
                        }
                        out_ptr0[static_cast<long>(x1 + (128L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))];
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp1 = c10::convert<long>(x2);
                        auto tmp2 = c10::convert<long>(1L + x1);
                        auto tmp3 = tmp1 < tmp2;
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp6 = tmp3 ? tmp4 : tmp5;
                        auto tmp7 = decltype(tmp0)(tmp0 + tmp6);
                        auto tmp9 = decltype(tmp7)(tmp7 - tmp8);
                        auto tmp10 = std::exp(tmp9);
                        in_out_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))] = tmp10;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
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


cpp_fused__softmax_clone_104 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (80L*x0) + (10240L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (80L*x1) + (2560L*x0)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2560L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2560L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(2560.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-05);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_107 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        auto tmp1 = static_cast<float>(0.11180339887498948);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr0 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        tmp0.store(out_ptr2 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_108 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(327680L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_109 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2560L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2560L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(2560.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-05);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp21.store(out_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_110 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1310720L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_111 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(327680L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2560L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(2560.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_112 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        auto tmp1 = static_cast<float>(0.11180339887498948);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr0 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_113 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))];
                            auto tmp1 = c10::convert<long>(x2);
                            auto tmp2 = c10::convert<long>(1L + x1);
                            auto tmp3 = tmp1 < tmp2;
                            auto tmp4 = static_cast<float>(0.0);
                            auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp6 = tmp3 ? tmp4 : tmp5;
                            auto tmp7 = decltype(tmp0)(tmp0 + tmp6);
                            tmp_acc0 = max_propagate_nan(tmp_acc0, tmp7);
                        }
                        out_ptr0[static_cast<long>(x1 + (128L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))];
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp1 = c10::convert<long>(x2);
                        auto tmp2 = c10::convert<long>(1L + x1);
                        auto tmp3 = tmp1 < tmp2;
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp6 = tmp3 ? tmp4 : tmp5;
                        auto tmp7 = decltype(tmp0)(tmp0 + tmp6);
                        auto tmp9 = decltype(tmp7)(tmp7 - tmp8);
                        auto tmp10 = std::exp(tmp9);
                        in_out_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))] = tmp10;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
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


cpp_fused__softmax_clone_114 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_115 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (80L*x0) + (10240L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (80L*x1) + (2560L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_116 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2560L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(2560.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_117 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        auto tmp1 = static_cast<float>(0.11180339887498948);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr0 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        tmp0.store(out_ptr2 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_118 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(327680L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_119 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2560L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2560L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(2560.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-05);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_120 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1310720L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_121 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2560L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2560L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(2560.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-05);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp21.store(out_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_122 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        auto tmp1 = static_cast<float>(0.11180339887498948);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr0 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_123 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))];
                            auto tmp1 = c10::convert<long>(x2);
                            auto tmp2 = c10::convert<long>(1L + x1);
                            auto tmp3 = tmp1 < tmp2;
                            auto tmp4 = static_cast<float>(0.0);
                            auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp6 = tmp3 ? tmp4 : tmp5;
                            auto tmp7 = decltype(tmp0)(tmp0 + tmp6);
                            tmp_acc0 = max_propagate_nan(tmp_acc0, tmp7);
                        }
                        out_ptr0[static_cast<long>(x1 + (128L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))];
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp1 = c10::convert<long>(x2);
                        auto tmp2 = c10::convert<long>(1L + x1);
                        auto tmp3 = tmp1 < tmp2;
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp6 = tmp3 ? tmp4 : tmp5;
                        auto tmp7 = decltype(tmp0)(tmp0 + tmp6);
                        auto tmp9 = decltype(tmp7)(tmp7 - tmp8);
                        auto tmp10 = std::exp(tmp9);
                        in_out_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))] = tmp10;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
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


cpp_fused__softmax_clone_124 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_125 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (80L*x0) + (10240L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (80L*x1) + (2560L*x0)));
                    }
                }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(327680L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2560L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(2560.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_127 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        auto tmp1 = static_cast<float>(0.11180339887498948);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr0 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        tmp0.store(out_ptr2 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_128 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(327680L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_129 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2560L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(2560.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_130 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1310720L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_131 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2560L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2560L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(2560.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-05);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_132 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        auto tmp1 = static_cast<float>(0.11180339887498948);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr0 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_133 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))];
                            auto tmp1 = c10::convert<long>(x2);
                            auto tmp2 = c10::convert<long>(1L + x1);
                            auto tmp3 = tmp1 < tmp2;
                            auto tmp4 = static_cast<float>(0.0);
                            auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp6 = tmp3 ? tmp4 : tmp5;
                            auto tmp7 = decltype(tmp0)(tmp0 + tmp6);
                            tmp_acc0 = max_propagate_nan(tmp_acc0, tmp7);
                        }
                        out_ptr0[static_cast<long>(x1 + (128L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))];
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp1 = c10::convert<long>(x2);
                        auto tmp2 = c10::convert<long>(1L + x1);
                        auto tmp3 = tmp1 < tmp2;
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp6 = tmp3 ? tmp4 : tmp5;
                        auto tmp7 = decltype(tmp0)(tmp0 + tmp6);
                        auto tmp9 = decltype(tmp7)(tmp7 - tmp8);
                        auto tmp10 = std::exp(tmp9);
                        in_out_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))] = tmp10;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
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


cpp_fused__softmax_clone_134 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (80L*x0) + (10240L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (80L*x1) + (2560L*x0)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2560L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2560L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(2560.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-05);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp21.store(out_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_137 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        auto tmp1 = static_cast<float>(0.11180339887498948);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr0 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        tmp0.store(out_ptr2 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_138 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(327680L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_139 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(327680L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2560L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(2560.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_140 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1310720L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_141 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2560L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(2560.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_142 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        auto tmp1 = static_cast<float>(0.11180339887498948);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr0 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_143 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))];
                            auto tmp1 = c10::convert<long>(x2);
                            auto tmp2 = c10::convert<long>(1L + x1);
                            auto tmp3 = tmp1 < tmp2;
                            auto tmp4 = static_cast<float>(0.0);
                            auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp6 = tmp3 ? tmp4 : tmp5;
                            auto tmp7 = decltype(tmp0)(tmp0 + tmp6);
                            tmp_acc0 = max_propagate_nan(tmp_acc0, tmp7);
                        }
                        out_ptr0[static_cast<long>(x1 + (128L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))];
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp1 = c10::convert<long>(x2);
                        auto tmp2 = c10::convert<long>(1L + x1);
                        auto tmp3 = tmp1 < tmp2;
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp6 = tmp3 ? tmp4 : tmp5;
                        auto tmp7 = decltype(tmp0)(tmp0 + tmp6);
                        auto tmp9 = decltype(tmp7)(tmp7 - tmp8);
                        auto tmp10 = std::exp(tmp9);
                        in_out_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))] = tmp10;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
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


cpp_fused__softmax_clone_144 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_145 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (80L*x0) + (10240L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (80L*x1) + (2560L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_146 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2560L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2560L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(2560.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-05);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_147 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        auto tmp1 = static_cast<float>(0.11180339887498948);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr0 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        tmp0.store(out_ptr2 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_148 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(327680L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_149 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2560L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2560L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(2560.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-05);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp21.store(out_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_150 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1310720L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_151 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(327680L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2560L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(2560.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_152 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        auto tmp1 = static_cast<float>(0.11180339887498948);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr0 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_153 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))];
                            auto tmp1 = c10::convert<long>(x2);
                            auto tmp2 = c10::convert<long>(1L + x1);
                            auto tmp3 = tmp1 < tmp2;
                            auto tmp4 = static_cast<float>(0.0);
                            auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp6 = tmp3 ? tmp4 : tmp5;
                            auto tmp7 = decltype(tmp0)(tmp0 + tmp6);
                            tmp_acc0 = max_propagate_nan(tmp_acc0, tmp7);
                        }
                        out_ptr0[static_cast<long>(x1 + (128L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))];
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp1 = c10::convert<long>(x2);
                        auto tmp2 = c10::convert<long>(1L + x1);
                        auto tmp3 = tmp1 < tmp2;
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp6 = tmp3 ? tmp4 : tmp5;
                        auto tmp7 = decltype(tmp0)(tmp0 + tmp6);
                        auto tmp9 = decltype(tmp7)(tmp7 - tmp8);
                        auto tmp10 = std::exp(tmp9);
                        in_out_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))] = tmp10;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
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


cpp_fused__softmax_clone_154 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_155 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (80L*x0) + (10240L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (80L*x1) + (2560L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_156 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2560L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(2560.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_157 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        auto tmp1 = static_cast<float>(0.11180339887498948);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr0 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        tmp0.store(out_ptr2 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_158 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(327680L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_159 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2560L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2560L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(2560.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-05);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_160 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1310720L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_161 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2560L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2560L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(2560.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-05);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp21.store(out_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_162 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        auto tmp1 = static_cast<float>(0.11180339887498948);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr0 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_163 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))];
                            auto tmp1 = c10::convert<long>(x2);
                            auto tmp2 = c10::convert<long>(1L + x1);
                            auto tmp3 = tmp1 < tmp2;
                            auto tmp4 = static_cast<float>(0.0);
                            auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp6 = tmp3 ? tmp4 : tmp5;
                            auto tmp7 = decltype(tmp0)(tmp0 + tmp6);
                            tmp_acc0 = max_propagate_nan(tmp_acc0, tmp7);
                        }
                        out_ptr0[static_cast<long>(x1 + (128L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))];
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp1 = c10::convert<long>(x2);
                        auto tmp2 = c10::convert<long>(1L + x1);
                        auto tmp3 = tmp1 < tmp2;
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp6 = tmp3 ? tmp4 : tmp5;
                        auto tmp7 = decltype(tmp0)(tmp0 + tmp6);
                        auto tmp9 = decltype(tmp7)(tmp7 - tmp8);
                        auto tmp10 = std::exp(tmp9);
                        in_out_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))] = tmp10;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
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


cpp_fused__softmax_clone_164 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_165 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (80L*x0) + (10240L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (80L*x1) + (2560L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_166 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(327680L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2560L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(2560.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_167 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        auto tmp1 = static_cast<float>(0.11180339887498948);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr0 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        tmp0.store(out_ptr2 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_168 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(327680L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_169 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2560L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(2560.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_170 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1310720L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_171 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2560L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2560L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(2560.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-05);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_172 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        auto tmp1 = static_cast<float>(0.11180339887498948);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr0 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_173 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))];
                            auto tmp1 = c10::convert<long>(x2);
                            auto tmp2 = c10::convert<long>(1L + x1);
                            auto tmp3 = tmp1 < tmp2;
                            auto tmp4 = static_cast<float>(0.0);
                            auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp6 = tmp3 ? tmp4 : tmp5;
                            auto tmp7 = decltype(tmp0)(tmp0 + tmp6);
                            tmp_acc0 = max_propagate_nan(tmp_acc0, tmp7);
                        }
                        out_ptr0[static_cast<long>(x1 + (128L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))];
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp1 = c10::convert<long>(x2);
                        auto tmp2 = c10::convert<long>(1L + x1);
                        auto tmp3 = tmp1 < tmp2;
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp6 = tmp3 ? tmp4 : tmp5;
                        auto tmp7 = decltype(tmp0)(tmp0 + tmp6);
                        auto tmp9 = decltype(tmp7)(tmp7 - tmp8);
                        auto tmp10 = std::exp(tmp9);
                        in_out_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))] = tmp10;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
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


cpp_fused__softmax_clone_174 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_175 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (80L*x0) + (10240L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (80L*x1) + (2560L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_176 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2560L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2560L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(2560.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-05);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp21.store(out_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_177 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        auto tmp1 = static_cast<float>(0.11180339887498948);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr0 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        tmp0.store(out_ptr2 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_178 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(327680L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_179 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(327680L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2560L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(2560.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_180 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1310720L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_181 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2560L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(2560.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_182 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        auto tmp1 = static_cast<float>(0.11180339887498948);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr0 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_183 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))];
                            auto tmp1 = c10::convert<long>(x2);
                            auto tmp2 = c10::convert<long>(1L + x1);
                            auto tmp3 = tmp1 < tmp2;
                            auto tmp4 = static_cast<float>(0.0);
                            auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp6 = tmp3 ? tmp4 : tmp5;
                            auto tmp7 = decltype(tmp0)(tmp0 + tmp6);
                            tmp_acc0 = max_propagate_nan(tmp_acc0, tmp7);
                        }
                        out_ptr0[static_cast<long>(x1 + (128L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))];
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp1 = c10::convert<long>(x2);
                        auto tmp2 = c10::convert<long>(1L + x1);
                        auto tmp3 = tmp1 < tmp2;
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp6 = tmp3 ? tmp4 : tmp5;
                        auto tmp7 = decltype(tmp0)(tmp0 + tmp6);
                        auto tmp9 = decltype(tmp7)(tmp7 - tmp8);
                        auto tmp10 = std::exp(tmp9);
                        in_out_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))] = tmp10;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
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


cpp_fused__softmax_clone_184 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_185 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (80L*x0) + (10240L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (80L*x1) + (2560L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_186 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2560L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2560L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(2560.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-05);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_187 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        auto tmp1 = static_cast<float>(0.11180339887498948);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr0 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        tmp0.store(out_ptr2 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_188 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(327680L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_189 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2560L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2560L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(2560.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-05);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp21.store(out_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_190 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1310720L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_191 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(327680L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2560L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(2560.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_192 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        auto tmp1 = static_cast<float>(0.11180339887498948);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr0 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_193 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))];
                            auto tmp1 = c10::convert<long>(x2);
                            auto tmp2 = c10::convert<long>(1L + x1);
                            auto tmp3 = tmp1 < tmp2;
                            auto tmp4 = static_cast<float>(0.0);
                            auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp6 = tmp3 ? tmp4 : tmp5;
                            auto tmp7 = decltype(tmp0)(tmp0 + tmp6);
                            tmp_acc0 = max_propagate_nan(tmp_acc0, tmp7);
                        }
                        out_ptr0[static_cast<long>(x1 + (128L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))];
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp1 = c10::convert<long>(x2);
                        auto tmp2 = c10::convert<long>(1L + x1);
                        auto tmp3 = tmp1 < tmp2;
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp6 = tmp3 ? tmp4 : tmp5;
                        auto tmp7 = decltype(tmp0)(tmp0 + tmp6);
                        auto tmp9 = decltype(tmp7)(tmp7 - tmp8);
                        auto tmp10 = std::exp(tmp9);
                        in_out_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))] = tmp10;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
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


cpp_fused__softmax_clone_194 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_195 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (80L*x0) + (10240L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (80L*x1) + (2560L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_196 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2560L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(2560.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_197 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        auto tmp1 = static_cast<float>(0.11180339887498948);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr0 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        tmp0.store(out_ptr2 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_198 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(327680L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_199 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2560L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2560L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(2560.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-05);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_200 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1310720L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_201 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2560L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2560L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(2560.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-05);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp21.store(out_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_202 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        auto tmp1 = static_cast<float>(0.11180339887498948);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr0 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_203 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))];
                            auto tmp1 = c10::convert<long>(x2);
                            auto tmp2 = c10::convert<long>(1L + x1);
                            auto tmp3 = tmp1 < tmp2;
                            auto tmp4 = static_cast<float>(0.0);
                            auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp6 = tmp3 ? tmp4 : tmp5;
                            auto tmp7 = decltype(tmp0)(tmp0 + tmp6);
                            tmp_acc0 = max_propagate_nan(tmp_acc0, tmp7);
                        }
                        out_ptr0[static_cast<long>(x1 + (128L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))];
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp1 = c10::convert<long>(x2);
                        auto tmp2 = c10::convert<long>(1L + x1);
                        auto tmp3 = tmp1 < tmp2;
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp6 = tmp3 ? tmp4 : tmp5;
                        auto tmp7 = decltype(tmp0)(tmp0 + tmp6);
                        auto tmp9 = decltype(tmp7)(tmp7 - tmp8);
                        auto tmp10 = std::exp(tmp9);
                        in_out_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))] = tmp10;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
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


cpp_fused__softmax_clone_204 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_205 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (80L*x0) + (10240L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (80L*x1) + (2560L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_206 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(327680L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2560L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(2560.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_207 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        auto tmp1 = static_cast<float>(0.11180339887498948);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr0 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        tmp0.store(out_ptr2 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_208 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(327680L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_209 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2560L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(2560.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_210 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1310720L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_211 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2560L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2560L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(2560.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-05);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_212 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        auto tmp1 = static_cast<float>(0.11180339887498948);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr0 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_213 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))];
                            auto tmp1 = c10::convert<long>(x2);
                            auto tmp2 = c10::convert<long>(1L + x1);
                            auto tmp3 = tmp1 < tmp2;
                            auto tmp4 = static_cast<float>(0.0);
                            auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp6 = tmp3 ? tmp4 : tmp5;
                            auto tmp7 = decltype(tmp0)(tmp0 + tmp6);
                            tmp_acc0 = max_propagate_nan(tmp_acc0, tmp7);
                        }
                        out_ptr0[static_cast<long>(x1 + (128L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))];
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp1 = c10::convert<long>(x2);
                        auto tmp2 = c10::convert<long>(1L + x1);
                        auto tmp3 = tmp1 < tmp2;
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp6 = tmp3 ? tmp4 : tmp5;
                        auto tmp7 = decltype(tmp0)(tmp0 + tmp6);
                        auto tmp9 = decltype(tmp7)(tmp7 - tmp8);
                        auto tmp10 = std::exp(tmp9);
                        in_out_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))] = tmp10;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
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


cpp_fused__softmax_clone_214 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_215 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (80L*x0) + (10240L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (80L*x1) + (2560L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_216 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2560L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2560L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(2560.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-05);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp21.store(out_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_217 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        auto tmp1 = static_cast<float>(0.11180339887498948);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr0 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        tmp0.store(out_ptr2 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_218 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(327680L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_219 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(327680L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2560L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(2560.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_220 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1310720L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_221 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2560L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(2560.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_222 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        auto tmp1 = static_cast<float>(0.11180339887498948);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr0 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_223 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))];
                            auto tmp1 = c10::convert<long>(x2);
                            auto tmp2 = c10::convert<long>(1L + x1);
                            auto tmp3 = tmp1 < tmp2;
                            auto tmp4 = static_cast<float>(0.0);
                            auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp6 = tmp3 ? tmp4 : tmp5;
                            auto tmp7 = decltype(tmp0)(tmp0 + tmp6);
                            tmp_acc0 = max_propagate_nan(tmp_acc0, tmp7);
                        }
                        out_ptr0[static_cast<long>(x1 + (128L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))];
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp1 = c10::convert<long>(x2);
                        auto tmp2 = c10::convert<long>(1L + x1);
                        auto tmp3 = tmp1 < tmp2;
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp6 = tmp3 ? tmp4 : tmp5;
                        auto tmp7 = decltype(tmp0)(tmp0 + tmp6);
                        auto tmp9 = decltype(tmp7)(tmp7 - tmp8);
                        auto tmp10 = std::exp(tmp9);
                        in_out_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))] = tmp10;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
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


cpp_fused__softmax_clone_224 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_225 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (80L*x0) + (10240L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (80L*x1) + (2560L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_226 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2560L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2560L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(2560.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-05);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_227 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        auto tmp1 = static_cast<float>(0.11180339887498948);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr0 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        tmp0.store(out_ptr2 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_228 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(327680L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_229 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2560L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2560L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(2560.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-05);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp21.store(out_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_230 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1310720L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_231 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(327680L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2560L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(2560.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_232 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        auto tmp1 = static_cast<float>(0.11180339887498948);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr0 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_233 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))];
                            auto tmp1 = c10::convert<long>(x2);
                            auto tmp2 = c10::convert<long>(1L + x1);
                            auto tmp3 = tmp1 < tmp2;
                            auto tmp4 = static_cast<float>(0.0);
                            auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp6 = tmp3 ? tmp4 : tmp5;
                            auto tmp7 = decltype(tmp0)(tmp0 + tmp6);
                            tmp_acc0 = max_propagate_nan(tmp_acc0, tmp7);
                        }
                        out_ptr0[static_cast<long>(x1 + (128L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))];
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp1 = c10::convert<long>(x2);
                        auto tmp2 = c10::convert<long>(1L + x1);
                        auto tmp3 = tmp1 < tmp2;
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp6 = tmp3 ? tmp4 : tmp5;
                        auto tmp7 = decltype(tmp0)(tmp0 + tmp6);
                        auto tmp9 = decltype(tmp7)(tmp7 - tmp8);
                        auto tmp10 = std::exp(tmp9);
                        in_out_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))] = tmp10;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
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


cpp_fused__softmax_clone_234 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_235 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (80L*x0) + (10240L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (80L*x1) + (2560L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_236 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2560L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(2560.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_237 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        auto tmp1 = static_cast<float>(0.11180339887498948);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr0 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        tmp0.store(out_ptr2 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_238 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(327680L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_239 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2560L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2560L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(2560.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-05);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_240 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1310720L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_241 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2560L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2560L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(2560.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-05);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp21.store(out_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_242 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        auto tmp1 = static_cast<float>(0.11180339887498948);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr0 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_243 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))];
                            auto tmp1 = c10::convert<long>(x2);
                            auto tmp2 = c10::convert<long>(1L + x1);
                            auto tmp3 = tmp1 < tmp2;
                            auto tmp4 = static_cast<float>(0.0);
                            auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp6 = tmp3 ? tmp4 : tmp5;
                            auto tmp7 = decltype(tmp0)(tmp0 + tmp6);
                            tmp_acc0 = max_propagate_nan(tmp_acc0, tmp7);
                        }
                        out_ptr0[static_cast<long>(x1 + (128L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))];
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp1 = c10::convert<long>(x2);
                        auto tmp2 = c10::convert<long>(1L + x1);
                        auto tmp3 = tmp1 < tmp2;
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp6 = tmp3 ? tmp4 : tmp5;
                        auto tmp7 = decltype(tmp0)(tmp0 + tmp6);
                        auto tmp9 = decltype(tmp7)(tmp7 - tmp8);
                        auto tmp10 = std::exp(tmp9);
                        in_out_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))] = tmp10;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
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


cpp_fused__softmax_clone_244 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(1L))
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_245 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (80L*x0) + (10240L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (80L*x1) + (2560L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_246 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(327680L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2560L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(2560.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_247 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        auto tmp1 = static_cast<float>(0.11180339887498948);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr0 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (80L*x0) + (2560L*x1)));
                        tmp0.store(out_ptr2 + static_cast<long>(x2 + (80L*x1) + (10240L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_248 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(327680L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_249 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2560L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(2560.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_250 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1310720L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_251 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2560L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2560L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(2560.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-05);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (2560L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__log_softmax_add_nll_loss_forward_252 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const long* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       long* out_ptr3)
{
    auto out_ptr2 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8008L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (8008L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                        auto tmp2 = tmp0 + tmp1;
                        tmp2.store(in_out_ptr0 + static_cast<long>(x1 + (8008L*x0)));
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp2);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8008L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (8008L*x0)));
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
        #pragma omp single
        {
            {
                {
                    float tmp_acc0 = 0;
                    long tmp_acc1 = 0;
                    for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr1[static_cast<long>(x0)];
                        auto tmp9 = out_ptr0[static_cast<long>(x0)];
                        auto tmp11 = out_ptr1[static_cast<long>(x0)];
                        auto tmp1 = static_cast<long>(-100);
                        auto tmp2 = tmp0 != tmp1;
                        auto tmp3 = static_cast<long>(0);
                        auto tmp4 = tmp2 ? tmp0 : tmp3;
                        auto tmp5 = decltype(tmp4)(tmp4 + 8008);
                        auto tmp6 = tmp4 < 0;
                        auto tmp7 = tmp6 ? tmp5 : tmp4;
                        TORCH_CHECK((0 <= tmp7) & (tmp7 < 8008L), "index out of bounds: 0 <= tmp7 < 8008L")
                        auto tmp8 = in_out_ptr0[static_cast<long>(tmp7 + (8008L*x0))];
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
                in_out_ptr1[static_cast<long>(0L)] = tmp3;
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1, arg473_1, arg474_1, arg475_1, arg476_1, arg477_1, arg478_1, arg479_1, arg480_1, arg481_1, arg482_1, arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, arg503_1, arg504_1, arg505_1, arg506_1, arg507_1, arg508_1, arg509_1, arg510_1, arg511_1, arg512_1, arg513_1, arg514_1, arg515_1, arg516_1, arg517_1, arg518_1, arg519_1, arg520_1, arg521_1, arg522_1, arg523_1, arg524_1, arg525_1, arg526_1, arg527_1, arg528_1, arg529_1, arg530_1, arg531_1, arg532_1, arg533_1, arg534_1, arg535_1, arg536_1, arg537_1, arg538_1, arg539_1, arg540_1, arg541_1, arg542_1, arg543_1, arg544_1, arg545_1, arg546_1, arg547_1, arg548_1, arg549_1, arg550_1, arg551_1, arg552_1, arg553_1, arg554_1, arg555_1, arg556_1, arg557_1, arg558_1, arg559_1, arg560_1, arg561_1, arg562_1, arg563_1, arg564_1, arg565_1, arg566_1, arg567_1, arg568_1, arg569_1, arg570_1, arg571_1, arg572_1, arg573_1, arg574_1, arg575_1, arg576_1, arg577_1, arg578_1, arg579_1, arg580_1, arg581_1, arg582_1, arg583_1, arg584_1, arg585_1, arg586_1, arg587_1, arg588_1, arg589_1, arg590_1, arg591_1, arg592_1, arg593_1, arg594_1, arg595_1, arg596_1, arg597_1, arg598_1, arg599_1, arg600_1, arg601_1, arg602_1, arg603_1, arg604_1, arg605_1, arg606_1, arg607_1, arg608_1, arg609_1, arg610_1, arg611_1, arg612_1, arg613_1, arg614_1, arg615_1, arg616_1, arg617_1, arg618_1, arg619_1, arg620_1, arg621_1, arg622_1, arg623_1, arg624_1, arg625_1, arg626_1, arg627_1, arg628_1, arg629_1, arg630_1, arg631_1, arg632_1, arg633_1, arg634_1, arg635_1, arg636_1, arg637_1, arg638_1, arg639_1, arg640_1, arg641_1, arg642_1, arg643_1, arg644_1, arg645_1, arg646_1, arg647_1, arg648_1, arg649_1, arg650_1, arg651_1, arg652_1, arg653_1, arg654_1, arg655_1, arg656_1, arg657_1, arg658_1, arg659_1, arg660_1, arg661_1, arg662_1, arg663_1, arg664_1, arg665_1, arg666_1, arg667_1 = args
    args.clear()
    assert_size_stride(arg0_1, (128, 2560), (2560, 1))
    assert_size_stride(arg1_1, (128, 2560), (2560, 1))
    assert_size_stride(arg2_1, (8008, 2560), (2560, 1))
    assert_size_stride(arg3_1, (2560, ), (1, ))
    assert_size_stride(arg4_1, (2560, ), (1, ))
    assert_size_stride(arg5_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg6_1, (2560, ), (1, ))
    assert_size_stride(arg7_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg8_1, (2560, ), (1, ))
    assert_size_stride(arg9_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg10_1, (2560, ), (1, ))
    assert_size_stride(arg11_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg12_1, (2560, ), (1, ))
    assert_size_stride(arg13_1, (2560, ), (1, ))
    assert_size_stride(arg14_1, (2560, ), (1, ))
    assert_size_stride(arg15_1, (10240, 2560), (2560, 1))
    assert_size_stride(arg16_1, (10240, ), (1, ))
    assert_size_stride(arg17_1, (2560, 10240), (10240, 1))
    assert_size_stride(arg18_1, (2560, ), (1, ))
    assert_size_stride(arg19_1, (2560, ), (1, ))
    assert_size_stride(arg20_1, (2560, ), (1, ))
    assert_size_stride(arg21_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg22_1, (2560, ), (1, ))
    assert_size_stride(arg23_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg24_1, (2560, ), (1, ))
    assert_size_stride(arg25_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg26_1, (2560, ), (1, ))
    assert_size_stride(arg27_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg28_1, (2560, ), (1, ))
    assert_size_stride(arg29_1, (2560, ), (1, ))
    assert_size_stride(arg30_1, (2560, ), (1, ))
    assert_size_stride(arg31_1, (10240, 2560), (2560, 1))
    assert_size_stride(arg32_1, (10240, ), (1, ))
    assert_size_stride(arg33_1, (2560, 10240), (10240, 1))
    assert_size_stride(arg34_1, (2560, ), (1, ))
    assert_size_stride(arg35_1, (2560, ), (1, ))
    assert_size_stride(arg36_1, (2560, ), (1, ))
    assert_size_stride(arg37_1, (2560, ), (1, ))
    assert_size_stride(arg38_1, (2560, ), (1, ))
    assert_size_stride(arg39_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg40_1, (2560, ), (1, ))
    assert_size_stride(arg41_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg42_1, (2560, ), (1, ))
    assert_size_stride(arg43_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg44_1, (2560, ), (1, ))
    assert_size_stride(arg45_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg46_1, (2560, ), (1, ))
    assert_size_stride(arg47_1, (2560, ), (1, ))
    assert_size_stride(arg48_1, (2560, ), (1, ))
    assert_size_stride(arg49_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg50_1, (2560, ), (1, ))
    assert_size_stride(arg51_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg52_1, (2560, ), (1, ))
    assert_size_stride(arg53_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg54_1, (2560, ), (1, ))
    assert_size_stride(arg55_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg56_1, (2560, ), (1, ))
    assert_size_stride(arg57_1, (2560, ), (1, ))
    assert_size_stride(arg58_1, (2560, ), (1, ))
    assert_size_stride(arg59_1, (10240, 2560), (2560, 1))
    assert_size_stride(arg60_1, (10240, ), (1, ))
    assert_size_stride(arg61_1, (2560, 10240), (10240, 1))
    assert_size_stride(arg62_1, (2560, ), (1, ))
    assert_size_stride(arg63_1, (2560, ), (1, ))
    assert_size_stride(arg64_1, (2560, ), (1, ))
    assert_size_stride(arg65_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg66_1, (2560, ), (1, ))
    assert_size_stride(arg67_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg68_1, (2560, ), (1, ))
    assert_size_stride(arg69_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg70_1, (2560, ), (1, ))
    assert_size_stride(arg71_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg72_1, (2560, ), (1, ))
    assert_size_stride(arg73_1, (2560, ), (1, ))
    assert_size_stride(arg74_1, (2560, ), (1, ))
    assert_size_stride(arg75_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg76_1, (2560, ), (1, ))
    assert_size_stride(arg77_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg78_1, (2560, ), (1, ))
    assert_size_stride(arg79_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg80_1, (2560, ), (1, ))
    assert_size_stride(arg81_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg82_1, (2560, ), (1, ))
    assert_size_stride(arg83_1, (2560, ), (1, ))
    assert_size_stride(arg84_1, (2560, ), (1, ))
    assert_size_stride(arg85_1, (10240, 2560), (2560, 1))
    assert_size_stride(arg86_1, (10240, ), (1, ))
    assert_size_stride(arg87_1, (2560, 10240), (10240, 1))
    assert_size_stride(arg88_1, (2560, ), (1, ))
    assert_size_stride(arg89_1, (2560, ), (1, ))
    assert_size_stride(arg90_1, (2560, ), (1, ))
    assert_size_stride(arg91_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg92_1, (2560, ), (1, ))
    assert_size_stride(arg93_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg94_1, (2560, ), (1, ))
    assert_size_stride(arg95_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg96_1, (2560, ), (1, ))
    assert_size_stride(arg97_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg98_1, (2560, ), (1, ))
    assert_size_stride(arg99_1, (2560, ), (1, ))
    assert_size_stride(arg100_1, (2560, ), (1, ))
    assert_size_stride(arg101_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg102_1, (2560, ), (1, ))
    assert_size_stride(arg103_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg104_1, (2560, ), (1, ))
    assert_size_stride(arg105_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg106_1, (2560, ), (1, ))
    assert_size_stride(arg107_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg108_1, (2560, ), (1, ))
    assert_size_stride(arg109_1, (2560, ), (1, ))
    assert_size_stride(arg110_1, (2560, ), (1, ))
    assert_size_stride(arg111_1, (10240, 2560), (2560, 1))
    assert_size_stride(arg112_1, (10240, ), (1, ))
    assert_size_stride(arg113_1, (2560, 10240), (10240, 1))
    assert_size_stride(arg114_1, (2560, ), (1, ))
    assert_size_stride(arg115_1, (2560, ), (1, ))
    assert_size_stride(arg116_1, (2560, ), (1, ))
    assert_size_stride(arg117_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg118_1, (2560, ), (1, ))
    assert_size_stride(arg119_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg120_1, (2560, ), (1, ))
    assert_size_stride(arg121_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg122_1, (2560, ), (1, ))
    assert_size_stride(arg123_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg124_1, (2560, ), (1, ))
    assert_size_stride(arg125_1, (2560, ), (1, ))
    assert_size_stride(arg126_1, (2560, ), (1, ))
    assert_size_stride(arg127_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg128_1, (2560, ), (1, ))
    assert_size_stride(arg129_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg130_1, (2560, ), (1, ))
    assert_size_stride(arg131_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg132_1, (2560, ), (1, ))
    assert_size_stride(arg133_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg134_1, (2560, ), (1, ))
    assert_size_stride(arg135_1, (2560, ), (1, ))
    assert_size_stride(arg136_1, (2560, ), (1, ))
    assert_size_stride(arg137_1, (10240, 2560), (2560, 1))
    assert_size_stride(arg138_1, (10240, ), (1, ))
    assert_size_stride(arg139_1, (2560, 10240), (10240, 1))
    assert_size_stride(arg140_1, (2560, ), (1, ))
    assert_size_stride(arg141_1, (2560, ), (1, ))
    assert_size_stride(arg142_1, (2560, ), (1, ))
    assert_size_stride(arg143_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg144_1, (2560, ), (1, ))
    assert_size_stride(arg145_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg146_1, (2560, ), (1, ))
    assert_size_stride(arg147_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg148_1, (2560, ), (1, ))
    assert_size_stride(arg149_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg150_1, (2560, ), (1, ))
    assert_size_stride(arg151_1, (2560, ), (1, ))
    assert_size_stride(arg152_1, (2560, ), (1, ))
    assert_size_stride(arg153_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg154_1, (2560, ), (1, ))
    assert_size_stride(arg155_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg156_1, (2560, ), (1, ))
    assert_size_stride(arg157_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg158_1, (2560, ), (1, ))
    assert_size_stride(arg159_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg160_1, (2560, ), (1, ))
    assert_size_stride(arg161_1, (2560, ), (1, ))
    assert_size_stride(arg162_1, (2560, ), (1, ))
    assert_size_stride(arg163_1, (10240, 2560), (2560, 1))
    assert_size_stride(arg164_1, (10240, ), (1, ))
    assert_size_stride(arg165_1, (2560, 10240), (10240, 1))
    assert_size_stride(arg166_1, (2560, ), (1, ))
    assert_size_stride(arg167_1, (2560, ), (1, ))
    assert_size_stride(arg168_1, (2560, ), (1, ))
    assert_size_stride(arg169_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg170_1, (2560, ), (1, ))
    assert_size_stride(arg171_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg172_1, (2560, ), (1, ))
    assert_size_stride(arg173_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg174_1, (2560, ), (1, ))
    assert_size_stride(arg175_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg176_1, (2560, ), (1, ))
    assert_size_stride(arg177_1, (2560, ), (1, ))
    assert_size_stride(arg178_1, (2560, ), (1, ))
    assert_size_stride(arg179_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg180_1, (2560, ), (1, ))
    assert_size_stride(arg181_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg182_1, (2560, ), (1, ))
    assert_size_stride(arg183_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg184_1, (2560, ), (1, ))
    assert_size_stride(arg185_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg186_1, (2560, ), (1, ))
    assert_size_stride(arg187_1, (2560, ), (1, ))
    assert_size_stride(arg188_1, (2560, ), (1, ))
    assert_size_stride(arg189_1, (10240, 2560), (2560, 1))
    assert_size_stride(arg190_1, (10240, ), (1, ))
    assert_size_stride(arg191_1, (2560, 10240), (10240, 1))
    assert_size_stride(arg192_1, (2560, ), (1, ))
    assert_size_stride(arg193_1, (2560, ), (1, ))
    assert_size_stride(arg194_1, (2560, ), (1, ))
    assert_size_stride(arg195_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg196_1, (2560, ), (1, ))
    assert_size_stride(arg197_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg198_1, (2560, ), (1, ))
    assert_size_stride(arg199_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg200_1, (2560, ), (1, ))
    assert_size_stride(arg201_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg202_1, (2560, ), (1, ))
    assert_size_stride(arg203_1, (2560, ), (1, ))
    assert_size_stride(arg204_1, (2560, ), (1, ))
    assert_size_stride(arg205_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg206_1, (2560, ), (1, ))
    assert_size_stride(arg207_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg208_1, (2560, ), (1, ))
    assert_size_stride(arg209_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg210_1, (2560, ), (1, ))
    assert_size_stride(arg211_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg212_1, (2560, ), (1, ))
    assert_size_stride(arg213_1, (2560, ), (1, ))
    assert_size_stride(arg214_1, (2560, ), (1, ))
    assert_size_stride(arg215_1, (10240, 2560), (2560, 1))
    assert_size_stride(arg216_1, (10240, ), (1, ))
    assert_size_stride(arg217_1, (2560, 10240), (10240, 1))
    assert_size_stride(arg218_1, (2560, ), (1, ))
    assert_size_stride(arg219_1, (2560, ), (1, ))
    assert_size_stride(arg220_1, (2560, ), (1, ))
    assert_size_stride(arg221_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg222_1, (2560, ), (1, ))
    assert_size_stride(arg223_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg224_1, (2560, ), (1, ))
    assert_size_stride(arg225_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg226_1, (2560, ), (1, ))
    assert_size_stride(arg227_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg228_1, (2560, ), (1, ))
    assert_size_stride(arg229_1, (2560, ), (1, ))
    assert_size_stride(arg230_1, (2560, ), (1, ))
    assert_size_stride(arg231_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg232_1, (2560, ), (1, ))
    assert_size_stride(arg233_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg234_1, (2560, ), (1, ))
    assert_size_stride(arg235_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg236_1, (2560, ), (1, ))
    assert_size_stride(arg237_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg238_1, (2560, ), (1, ))
    assert_size_stride(arg239_1, (2560, ), (1, ))
    assert_size_stride(arg240_1, (2560, ), (1, ))
    assert_size_stride(arg241_1, (10240, 2560), (2560, 1))
    assert_size_stride(arg242_1, (10240, ), (1, ))
    assert_size_stride(arg243_1, (2560, 10240), (10240, 1))
    assert_size_stride(arg244_1, (2560, ), (1, ))
    assert_size_stride(arg245_1, (2560, ), (1, ))
    assert_size_stride(arg246_1, (2560, ), (1, ))
    assert_size_stride(arg247_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg248_1, (2560, ), (1, ))
    assert_size_stride(arg249_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg250_1, (2560, ), (1, ))
    assert_size_stride(arg251_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg252_1, (2560, ), (1, ))
    assert_size_stride(arg253_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg254_1, (2560, ), (1, ))
    assert_size_stride(arg255_1, (2560, ), (1, ))
    assert_size_stride(arg256_1, (2560, ), (1, ))
    assert_size_stride(arg257_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg258_1, (2560, ), (1, ))
    assert_size_stride(arg259_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg260_1, (2560, ), (1, ))
    assert_size_stride(arg261_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg262_1, (2560, ), (1, ))
    assert_size_stride(arg263_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg264_1, (2560, ), (1, ))
    assert_size_stride(arg265_1, (2560, ), (1, ))
    assert_size_stride(arg266_1, (2560, ), (1, ))
    assert_size_stride(arg267_1, (10240, 2560), (2560, 1))
    assert_size_stride(arg268_1, (10240, ), (1, ))
    assert_size_stride(arg269_1, (2560, 10240), (10240, 1))
    assert_size_stride(arg270_1, (2560, ), (1, ))
    assert_size_stride(arg271_1, (2560, ), (1, ))
    assert_size_stride(arg272_1, (2560, ), (1, ))
    assert_size_stride(arg273_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg274_1, (2560, ), (1, ))
    assert_size_stride(arg275_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg276_1, (2560, ), (1, ))
    assert_size_stride(arg277_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg278_1, (2560, ), (1, ))
    assert_size_stride(arg279_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg280_1, (2560, ), (1, ))
    assert_size_stride(arg281_1, (2560, ), (1, ))
    assert_size_stride(arg282_1, (2560, ), (1, ))
    assert_size_stride(arg283_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg284_1, (2560, ), (1, ))
    assert_size_stride(arg285_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg286_1, (2560, ), (1, ))
    assert_size_stride(arg287_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg288_1, (2560, ), (1, ))
    assert_size_stride(arg289_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg290_1, (2560, ), (1, ))
    assert_size_stride(arg291_1, (2560, ), (1, ))
    assert_size_stride(arg292_1, (2560, ), (1, ))
    assert_size_stride(arg293_1, (10240, 2560), (2560, 1))
    assert_size_stride(arg294_1, (10240, ), (1, ))
    assert_size_stride(arg295_1, (2560, 10240), (10240, 1))
    assert_size_stride(arg296_1, (2560, ), (1, ))
    assert_size_stride(arg297_1, (2560, ), (1, ))
    assert_size_stride(arg298_1, (2560, ), (1, ))
    assert_size_stride(arg299_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg300_1, (2560, ), (1, ))
    assert_size_stride(arg301_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg302_1, (2560, ), (1, ))
    assert_size_stride(arg303_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg304_1, (2560, ), (1, ))
    assert_size_stride(arg305_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg306_1, (2560, ), (1, ))
    assert_size_stride(arg307_1, (2560, ), (1, ))
    assert_size_stride(arg308_1, (2560, ), (1, ))
    assert_size_stride(arg309_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg310_1, (2560, ), (1, ))
    assert_size_stride(arg311_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg312_1, (2560, ), (1, ))
    assert_size_stride(arg313_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg314_1, (2560, ), (1, ))
    assert_size_stride(arg315_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg316_1, (2560, ), (1, ))
    assert_size_stride(arg317_1, (2560, ), (1, ))
    assert_size_stride(arg318_1, (2560, ), (1, ))
    assert_size_stride(arg319_1, (10240, 2560), (2560, 1))
    assert_size_stride(arg320_1, (10240, ), (1, ))
    assert_size_stride(arg321_1, (2560, 10240), (10240, 1))
    assert_size_stride(arg322_1, (2560, ), (1, ))
    assert_size_stride(arg323_1, (2560, ), (1, ))
    assert_size_stride(arg324_1, (2560, ), (1, ))
    assert_size_stride(arg325_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg326_1, (2560, ), (1, ))
    assert_size_stride(arg327_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg328_1, (2560, ), (1, ))
    assert_size_stride(arg329_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg330_1, (2560, ), (1, ))
    assert_size_stride(arg331_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg332_1, (2560, ), (1, ))
    assert_size_stride(arg333_1, (2560, ), (1, ))
    assert_size_stride(arg334_1, (2560, ), (1, ))
    assert_size_stride(arg335_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg336_1, (2560, ), (1, ))
    assert_size_stride(arg337_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg338_1, (2560, ), (1, ))
    assert_size_stride(arg339_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg340_1, (2560, ), (1, ))
    assert_size_stride(arg341_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg342_1, (2560, ), (1, ))
    assert_size_stride(arg343_1, (2560, ), (1, ))
    assert_size_stride(arg344_1, (2560, ), (1, ))
    assert_size_stride(arg345_1, (10240, 2560), (2560, 1))
    assert_size_stride(arg346_1, (10240, ), (1, ))
    assert_size_stride(arg347_1, (2560, 10240), (10240, 1))
    assert_size_stride(arg348_1, (2560, ), (1, ))
    assert_size_stride(arg349_1, (2560, ), (1, ))
    assert_size_stride(arg350_1, (2560, ), (1, ))
    assert_size_stride(arg351_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg352_1, (2560, ), (1, ))
    assert_size_stride(arg353_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg354_1, (2560, ), (1, ))
    assert_size_stride(arg355_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg356_1, (2560, ), (1, ))
    assert_size_stride(arg357_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg358_1, (2560, ), (1, ))
    assert_size_stride(arg359_1, (2560, ), (1, ))
    assert_size_stride(arg360_1, (2560, ), (1, ))
    assert_size_stride(arg361_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg362_1, (2560, ), (1, ))
    assert_size_stride(arg363_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg364_1, (2560, ), (1, ))
    assert_size_stride(arg365_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg366_1, (2560, ), (1, ))
    assert_size_stride(arg367_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg368_1, (2560, ), (1, ))
    assert_size_stride(arg369_1, (2560, ), (1, ))
    assert_size_stride(arg370_1, (2560, ), (1, ))
    assert_size_stride(arg371_1, (10240, 2560), (2560, 1))
    assert_size_stride(arg372_1, (10240, ), (1, ))
    assert_size_stride(arg373_1, (2560, 10240), (10240, 1))
    assert_size_stride(arg374_1, (2560, ), (1, ))
    assert_size_stride(arg375_1, (2560, ), (1, ))
    assert_size_stride(arg376_1, (2560, ), (1, ))
    assert_size_stride(arg377_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg378_1, (2560, ), (1, ))
    assert_size_stride(arg379_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg380_1, (2560, ), (1, ))
    assert_size_stride(arg381_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg382_1, (2560, ), (1, ))
    assert_size_stride(arg383_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg384_1, (2560, ), (1, ))
    assert_size_stride(arg385_1, (2560, ), (1, ))
    assert_size_stride(arg386_1, (2560, ), (1, ))
    assert_size_stride(arg387_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg388_1, (2560, ), (1, ))
    assert_size_stride(arg389_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg390_1, (2560, ), (1, ))
    assert_size_stride(arg391_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg392_1, (2560, ), (1, ))
    assert_size_stride(arg393_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg394_1, (2560, ), (1, ))
    assert_size_stride(arg395_1, (2560, ), (1, ))
    assert_size_stride(arg396_1, (2560, ), (1, ))
    assert_size_stride(arg397_1, (10240, 2560), (2560, 1))
    assert_size_stride(arg398_1, (10240, ), (1, ))
    assert_size_stride(arg399_1, (2560, 10240), (10240, 1))
    assert_size_stride(arg400_1, (2560, ), (1, ))
    assert_size_stride(arg401_1, (2560, ), (1, ))
    assert_size_stride(arg402_1, (2560, ), (1, ))
    assert_size_stride(arg403_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg404_1, (2560, ), (1, ))
    assert_size_stride(arg405_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg406_1, (2560, ), (1, ))
    assert_size_stride(arg407_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg408_1, (2560, ), (1, ))
    assert_size_stride(arg409_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg410_1, (2560, ), (1, ))
    assert_size_stride(arg411_1, (2560, ), (1, ))
    assert_size_stride(arg412_1, (2560, ), (1, ))
    assert_size_stride(arg413_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg414_1, (2560, ), (1, ))
    assert_size_stride(arg415_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg416_1, (2560, ), (1, ))
    assert_size_stride(arg417_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg418_1, (2560, ), (1, ))
    assert_size_stride(arg419_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg420_1, (2560, ), (1, ))
    assert_size_stride(arg421_1, (2560, ), (1, ))
    assert_size_stride(arg422_1, (2560, ), (1, ))
    assert_size_stride(arg423_1, (10240, 2560), (2560, 1))
    assert_size_stride(arg424_1, (10240, ), (1, ))
    assert_size_stride(arg425_1, (2560, 10240), (10240, 1))
    assert_size_stride(arg426_1, (2560, ), (1, ))
    assert_size_stride(arg427_1, (2560, ), (1, ))
    assert_size_stride(arg428_1, (2560, ), (1, ))
    assert_size_stride(arg429_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg430_1, (2560, ), (1, ))
    assert_size_stride(arg431_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg432_1, (2560, ), (1, ))
    assert_size_stride(arg433_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg434_1, (2560, ), (1, ))
    assert_size_stride(arg435_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg436_1, (2560, ), (1, ))
    assert_size_stride(arg437_1, (2560, ), (1, ))
    assert_size_stride(arg438_1, (2560, ), (1, ))
    assert_size_stride(arg439_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg440_1, (2560, ), (1, ))
    assert_size_stride(arg441_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg442_1, (2560, ), (1, ))
    assert_size_stride(arg443_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg444_1, (2560, ), (1, ))
    assert_size_stride(arg445_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg446_1, (2560, ), (1, ))
    assert_size_stride(arg447_1, (2560, ), (1, ))
    assert_size_stride(arg448_1, (2560, ), (1, ))
    assert_size_stride(arg449_1, (10240, 2560), (2560, 1))
    assert_size_stride(arg450_1, (10240, ), (1, ))
    assert_size_stride(arg451_1, (2560, 10240), (10240, 1))
    assert_size_stride(arg452_1, (2560, ), (1, ))
    assert_size_stride(arg453_1, (2560, ), (1, ))
    assert_size_stride(arg454_1, (2560, ), (1, ))
    assert_size_stride(arg455_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg456_1, (2560, ), (1, ))
    assert_size_stride(arg457_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg458_1, (2560, ), (1, ))
    assert_size_stride(arg459_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg460_1, (2560, ), (1, ))
    assert_size_stride(arg461_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg462_1, (2560, ), (1, ))
    assert_size_stride(arg463_1, (2560, ), (1, ))
    assert_size_stride(arg464_1, (2560, ), (1, ))
    assert_size_stride(arg465_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg466_1, (2560, ), (1, ))
    assert_size_stride(arg467_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg468_1, (2560, ), (1, ))
    assert_size_stride(arg469_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg470_1, (2560, ), (1, ))
    assert_size_stride(arg471_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg472_1, (2560, ), (1, ))
    assert_size_stride(arg473_1, (2560, ), (1, ))
    assert_size_stride(arg474_1, (2560, ), (1, ))
    assert_size_stride(arg475_1, (10240, 2560), (2560, 1))
    assert_size_stride(arg476_1, (10240, ), (1, ))
    assert_size_stride(arg477_1, (2560, 10240), (10240, 1))
    assert_size_stride(arg478_1, (2560, ), (1, ))
    assert_size_stride(arg479_1, (2560, ), (1, ))
    assert_size_stride(arg480_1, (2560, ), (1, ))
    assert_size_stride(arg481_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg482_1, (2560, ), (1, ))
    assert_size_stride(arg483_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg484_1, (2560, ), (1, ))
    assert_size_stride(arg485_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg486_1, (2560, ), (1, ))
    assert_size_stride(arg487_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg488_1, (2560, ), (1, ))
    assert_size_stride(arg489_1, (2560, ), (1, ))
    assert_size_stride(arg490_1, (2560, ), (1, ))
    assert_size_stride(arg491_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg492_1, (2560, ), (1, ))
    assert_size_stride(arg493_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg494_1, (2560, ), (1, ))
    assert_size_stride(arg495_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg496_1, (2560, ), (1, ))
    assert_size_stride(arg497_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg498_1, (2560, ), (1, ))
    assert_size_stride(arg499_1, (2560, ), (1, ))
    assert_size_stride(arg500_1, (2560, ), (1, ))
    assert_size_stride(arg501_1, (10240, 2560), (2560, 1))
    assert_size_stride(arg502_1, (10240, ), (1, ))
    assert_size_stride(arg503_1, (2560, 10240), (10240, 1))
    assert_size_stride(arg504_1, (2560, ), (1, ))
    assert_size_stride(arg505_1, (2560, ), (1, ))
    assert_size_stride(arg506_1, (2560, ), (1, ))
    assert_size_stride(arg507_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg508_1, (2560, ), (1, ))
    assert_size_stride(arg509_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg510_1, (2560, ), (1, ))
    assert_size_stride(arg511_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg512_1, (2560, ), (1, ))
    assert_size_stride(arg513_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg514_1, (2560, ), (1, ))
    assert_size_stride(arg515_1, (2560, ), (1, ))
    assert_size_stride(arg516_1, (2560, ), (1, ))
    assert_size_stride(arg517_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg518_1, (2560, ), (1, ))
    assert_size_stride(arg519_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg520_1, (2560, ), (1, ))
    assert_size_stride(arg521_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg522_1, (2560, ), (1, ))
    assert_size_stride(arg523_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg524_1, (2560, ), (1, ))
    assert_size_stride(arg525_1, (2560, ), (1, ))
    assert_size_stride(arg526_1, (2560, ), (1, ))
    assert_size_stride(arg527_1, (10240, 2560), (2560, 1))
    assert_size_stride(arg528_1, (10240, ), (1, ))
    assert_size_stride(arg529_1, (2560, 10240), (10240, 1))
    assert_size_stride(arg530_1, (2560, ), (1, ))
    assert_size_stride(arg531_1, (2560, ), (1, ))
    assert_size_stride(arg532_1, (2560, ), (1, ))
    assert_size_stride(arg533_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg534_1, (2560, ), (1, ))
    assert_size_stride(arg535_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg536_1, (2560, ), (1, ))
    assert_size_stride(arg537_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg538_1, (2560, ), (1, ))
    assert_size_stride(arg539_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg540_1, (2560, ), (1, ))
    assert_size_stride(arg541_1, (2560, ), (1, ))
    assert_size_stride(arg542_1, (2560, ), (1, ))
    assert_size_stride(arg543_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg544_1, (2560, ), (1, ))
    assert_size_stride(arg545_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg546_1, (2560, ), (1, ))
    assert_size_stride(arg547_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg548_1, (2560, ), (1, ))
    assert_size_stride(arg549_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg550_1, (2560, ), (1, ))
    assert_size_stride(arg551_1, (2560, ), (1, ))
    assert_size_stride(arg552_1, (2560, ), (1, ))
    assert_size_stride(arg553_1, (10240, 2560), (2560, 1))
    assert_size_stride(arg554_1, (10240, ), (1, ))
    assert_size_stride(arg555_1, (2560, 10240), (10240, 1))
    assert_size_stride(arg556_1, (2560, ), (1, ))
    assert_size_stride(arg557_1, (2560, ), (1, ))
    assert_size_stride(arg558_1, (2560, ), (1, ))
    assert_size_stride(arg559_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg560_1, (2560, ), (1, ))
    assert_size_stride(arg561_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg562_1, (2560, ), (1, ))
    assert_size_stride(arg563_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg564_1, (2560, ), (1, ))
    assert_size_stride(arg565_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg566_1, (2560, ), (1, ))
    assert_size_stride(arg567_1, (2560, ), (1, ))
    assert_size_stride(arg568_1, (2560, ), (1, ))
    assert_size_stride(arg569_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg570_1, (2560, ), (1, ))
    assert_size_stride(arg571_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg572_1, (2560, ), (1, ))
    assert_size_stride(arg573_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg574_1, (2560, ), (1, ))
    assert_size_stride(arg575_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg576_1, (2560, ), (1, ))
    assert_size_stride(arg577_1, (2560, ), (1, ))
    assert_size_stride(arg578_1, (2560, ), (1, ))
    assert_size_stride(arg579_1, (10240, 2560), (2560, 1))
    assert_size_stride(arg580_1, (10240, ), (1, ))
    assert_size_stride(arg581_1, (2560, 10240), (10240, 1))
    assert_size_stride(arg582_1, (2560, ), (1, ))
    assert_size_stride(arg583_1, (2560, ), (1, ))
    assert_size_stride(arg584_1, (2560, ), (1, ))
    assert_size_stride(arg585_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg586_1, (2560, ), (1, ))
    assert_size_stride(arg587_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg588_1, (2560, ), (1, ))
    assert_size_stride(arg589_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg590_1, (2560, ), (1, ))
    assert_size_stride(arg591_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg592_1, (2560, ), (1, ))
    assert_size_stride(arg593_1, (2560, ), (1, ))
    assert_size_stride(arg594_1, (2560, ), (1, ))
    assert_size_stride(arg595_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg596_1, (2560, ), (1, ))
    assert_size_stride(arg597_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg598_1, (2560, ), (1, ))
    assert_size_stride(arg599_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg600_1, (2560, ), (1, ))
    assert_size_stride(arg601_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg602_1, (2560, ), (1, ))
    assert_size_stride(arg603_1, (2560, ), (1, ))
    assert_size_stride(arg604_1, (2560, ), (1, ))
    assert_size_stride(arg605_1, (10240, 2560), (2560, 1))
    assert_size_stride(arg606_1, (10240, ), (1, ))
    assert_size_stride(arg607_1, (2560, 10240), (10240, 1))
    assert_size_stride(arg608_1, (2560, ), (1, ))
    assert_size_stride(arg609_1, (2560, ), (1, ))
    assert_size_stride(arg610_1, (2560, ), (1, ))
    assert_size_stride(arg611_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg612_1, (2560, ), (1, ))
    assert_size_stride(arg613_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg614_1, (2560, ), (1, ))
    assert_size_stride(arg615_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg616_1, (2560, ), (1, ))
    assert_size_stride(arg617_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg618_1, (2560, ), (1, ))
    assert_size_stride(arg619_1, (2560, ), (1, ))
    assert_size_stride(arg620_1, (2560, ), (1, ))
    assert_size_stride(arg621_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg622_1, (2560, ), (1, ))
    assert_size_stride(arg623_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg624_1, (2560, ), (1, ))
    assert_size_stride(arg625_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg626_1, (2560, ), (1, ))
    assert_size_stride(arg627_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg628_1, (2560, ), (1, ))
    assert_size_stride(arg629_1, (2560, ), (1, ))
    assert_size_stride(arg630_1, (2560, ), (1, ))
    assert_size_stride(arg631_1, (10240, 2560), (2560, 1))
    assert_size_stride(arg632_1, (10240, ), (1, ))
    assert_size_stride(arg633_1, (2560, 10240), (10240, 1))
    assert_size_stride(arg634_1, (2560, ), (1, ))
    assert_size_stride(arg635_1, (2560, ), (1, ))
    assert_size_stride(arg636_1, (2560, ), (1, ))
    assert_size_stride(arg637_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg638_1, (2560, ), (1, ))
    assert_size_stride(arg639_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg640_1, (2560, ), (1, ))
    assert_size_stride(arg641_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg642_1, (2560, ), (1, ))
    assert_size_stride(arg643_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg644_1, (2560, ), (1, ))
    assert_size_stride(arg645_1, (2560, ), (1, ))
    assert_size_stride(arg646_1, (2560, ), (1, ))
    assert_size_stride(arg647_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg648_1, (2560, ), (1, ))
    assert_size_stride(arg649_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg650_1, (2560, ), (1, ))
    assert_size_stride(arg651_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg652_1, (2560, ), (1, ))
    assert_size_stride(arg653_1, (2560, 2560), (2560, 1))
    assert_size_stride(arg654_1, (2560, ), (1, ))
    assert_size_stride(arg655_1, (2560, ), (1, ))
    assert_size_stride(arg656_1, (2560, ), (1, ))
    assert_size_stride(arg657_1, (10240, 2560), (2560, 1))
    assert_size_stride(arg658_1, (10240, ), (1, ))
    assert_size_stride(arg659_1, (2560, 10240), (10240, 1))
    assert_size_stride(arg660_1, (2560, ), (1, ))
    assert_size_stride(arg661_1, (2560, ), (1, ))
    assert_size_stride(arg662_1, (2560, ), (1, ))
    assert_size_stride(arg663_1, (8008, 2560), (2560, 1))
    assert_size_stride(arg664_1, (1, 8008), (8008, 1))
    assert_size_stride(arg665_1, (1, 128), (128, 1))
    assert_size_stride(arg666_1, (1, 128), (128, 1))
    assert_size_stride(arg667_1, (1, 128), (128, 1))
    buf0 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf1 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf3 = empty((1, 128, 2560), device='cpu', dtype=torch.float32)
    cpp_fused_add_arange_embedding_mul_native_layer_norm_0(c_void_p(arg667_1.data_ptr()), c_void_p(arg2_1.data_ptr()), c_void_p(arg0_1.data_ptr()), c_void_p(arg3_1.data_ptr()), c_void_p(arg4_1.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()), c_void_p(buf3.data_ptr()))
    del arg3_1
    del arg4_1
    buf4 = empty((128, 2560), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_encoder_layers_0_self_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg6_1, reinterpret_tensor(buf3, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg5_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf4)
    del arg5_1
    del arg6_1
    buf5 = empty((128, 2560), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_encoder_layers_0_self_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg8_1, reinterpret_tensor(buf3, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg7_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf5)
    del arg7_1
    del arg8_1
    buf6 = empty((128, 2560), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_encoder_layers_0_self_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg10_1, reinterpret_tensor(buf3, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg9_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf6)
    del arg10_1
    del arg9_1
    buf7 = reinterpret_tensor(buf3, (1, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf3  # reuse
    buf8 = empty((1, 32, 128, 80), device='cpu', dtype=torch.float32)
    buf9 = empty((1, 32, 128, 80), device='cpu', dtype=torch.float32)
    cpp_fused_clone_1(c_void_p(buf4.data_ptr()), c_void_p(buf5.data_ptr()), c_void_p(buf6.data_ptr()), c_void_p(buf7.data_ptr()), c_void_p(buf8.data_ptr()), c_void_p(buf9.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf10 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf7, (1, 32, 128, 80), (0, 10240, 80, 1), 0), reinterpret_tensor(buf8, (1, 32, 128, 80), (0, 10240, 80, 1), 0), reinterpret_tensor(buf9, (1, 32, 128, 80), (0, 10240, 80, 1), 0), scale=1.0)
    buf11 = buf10[0]
    del buf10
    buf18 = reinterpret_tensor(buf11, (1, 128, 32, 80), (327680, 2560, 80, 1), 0); del buf11  # reuse
    cpp_fused_clone_2(c_void_p(buf18.data_ptr()))
    buf19 = reinterpret_tensor(buf9, (128, 2560), (2560, 1), 0); del buf9  # reuse
    # Source Nodes: [hidden_states_3], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg12_1, reinterpret_tensor(buf18, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg11_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf19)
    del arg11_1
    del arg12_1
    buf20 = buf1; del buf1  # reuse
    buf21 = buf0; del buf0  # reuse
    buf58 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf59 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf23 = reinterpret_tensor(buf18, (1, 128, 2560), (327680, 2560, 1), 0); del buf18  # reuse
    cpp_fused_add_arange_embedding_mul_native_layer_norm_3(c_void_p(arg667_1.data_ptr()), c_void_p(arg2_1.data_ptr()), c_void_p(arg0_1.data_ptr()), c_void_p(buf19.data_ptr()), c_void_p(arg666_1.data_ptr()), c_void_p(arg1_1.data_ptr()), c_void_p(arg13_1.data_ptr()), c_void_p(arg14_1.data_ptr()), c_void_p(buf20.data_ptr()), c_void_p(buf21.data_ptr()), c_void_p(buf58.data_ptr()), c_void_p(buf59.data_ptr()), c_void_p(buf23.data_ptr()))
    del arg13_1
    del arg14_1
    buf24 = empty((128, 10240), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_encoder_layers_0_fc1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg16_1, reinterpret_tensor(buf23, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg15_1, (2560, 10240), (1, 2560), 0), alpha=1, beta=1, out=buf24)
    del arg15_1
    del arg16_1
    buf25 = reinterpret_tensor(buf24, (1, 128, 10240), (1310720, 10240, 1), 0); del buf24  # reuse
    cpp_fused_gelu_4(c_void_p(buf25.data_ptr()))
    buf26 = reinterpret_tensor(buf23, (128, 2560), (2560, 1), 0); del buf23  # reuse
    # Source Nodes: [hidden_states_9], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg18_1, reinterpret_tensor(buf25, (128, 10240), (10240, 1), 0), reinterpret_tensor(arg17_1, (10240, 2560), (1, 10240), 0), alpha=1, beta=1, out=buf26)
    del arg17_1
    del arg18_1
    buf27 = reinterpret_tensor(buf26, (1, 128, 2560), (327680, 2560, 1), 0); del buf26  # reuse
    buf61 = reinterpret_tensor(buf8, (1, 128, 2560), (327680, 2560, 1), 0); del buf8  # reuse
    buf28 = buf21; del buf21  # reuse
    buf29 = buf20; del buf20  # reuse
    buf31 = reinterpret_tensor(buf7, (1, 128, 2560), (327680, 2560, 1), 0); del buf7  # reuse
    cpp_fused_add_arange_embedding_mul_native_layer_norm_5(c_void_p(buf27.data_ptr()), c_void_p(arg667_1.data_ptr()), c_void_p(arg2_1.data_ptr()), c_void_p(arg0_1.data_ptr()), c_void_p(buf19.data_ptr()), c_void_p(arg666_1.data_ptr()), c_void_p(arg1_1.data_ptr()), c_void_p(buf58.data_ptr()), c_void_p(buf59.data_ptr()), c_void_p(arg37_1.data_ptr()), c_void_p(arg38_1.data_ptr()), c_void_p(arg19_1.data_ptr()), c_void_p(arg20_1.data_ptr()), c_void_p(buf61.data_ptr()), c_void_p(buf28.data_ptr()), c_void_p(buf29.data_ptr()), c_void_p(buf31.data_ptr()))
    del arg0_1
    del arg19_1
    del arg20_1
    del arg37_1
    del arg38_1
    del arg667_1
    buf32 = buf19; del buf19  # reuse
    # Source Nodes: [l__mod___model_encoder_layers_1_self_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg22_1, reinterpret_tensor(buf31, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg21_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf32)
    del arg21_1
    del arg22_1
    buf33 = buf6; del buf6  # reuse
    # Source Nodes: [l__mod___model_encoder_layers_1_self_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg24_1, reinterpret_tensor(buf31, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg23_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf33)
    del arg23_1
    del arg24_1
    buf34 = buf5; del buf5  # reuse
    # Source Nodes: [l__mod___model_encoder_layers_1_self_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg26_1, reinterpret_tensor(buf31, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg25_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf34)
    del arg25_1
    del arg26_1
    buf35 = reinterpret_tensor(buf31, (1, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf31  # reuse
    buf36 = reinterpret_tensor(buf4, (1, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf4  # reuse
    buf37 = empty((1, 32, 128, 80), device='cpu', dtype=torch.float32)
    cpp_fused_clone_6(c_void_p(buf32.data_ptr()), c_void_p(buf33.data_ptr()), c_void_p(buf34.data_ptr()), c_void_p(buf35.data_ptr()), c_void_p(buf36.data_ptr()), c_void_p(buf37.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf38 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf35, (1, 32, 128, 80), (0, 10240, 80, 1), 0), reinterpret_tensor(buf36, (1, 32, 128, 80), (0, 10240, 80, 1), 0), reinterpret_tensor(buf37, (1, 32, 128, 80), (0, 10240, 80, 1), 0), scale=1.0)
    buf39 = buf38[0]
    del buf38
    buf46 = reinterpret_tensor(buf39, (1, 128, 32, 80), (327680, 2560, 80, 1), 0); del buf39  # reuse
    cpp_fused_clone_7(c_void_p(buf46.data_ptr()))
    buf47 = reinterpret_tensor(buf37, (128, 2560), (2560, 1), 0); del buf37  # reuse
    # Source Nodes: [hidden_states_14], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg28_1, reinterpret_tensor(buf46, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg27_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf47)
    del arg27_1
    del arg28_1
    buf48 = buf59; del buf59  # reuse
    buf49 = buf58; del buf58  # reuse
    buf51 = reinterpret_tensor(buf46, (1, 128, 2560), (327680, 2560, 1), 0); del buf46  # reuse
    cpp_fused_add_native_layer_norm_8(c_void_p(buf27.data_ptr()), c_void_p(buf47.data_ptr()), c_void_p(arg29_1.data_ptr()), c_void_p(arg30_1.data_ptr()), c_void_p(buf48.data_ptr()), c_void_p(buf49.data_ptr()), c_void_p(buf51.data_ptr()))
    del arg29_1
    del arg30_1
    buf52 = reinterpret_tensor(buf25, (128, 10240), (10240, 1), 0); del buf25  # reuse
    # Source Nodes: [l__mod___model_encoder_layers_1_fc1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg32_1, reinterpret_tensor(buf51, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg31_1, (2560, 10240), (1, 2560), 0), alpha=1, beta=1, out=buf52)
    del arg31_1
    del arg32_1
    buf53 = reinterpret_tensor(buf52, (1, 128, 10240), (1310720, 10240, 1), 0); del buf52  # reuse
    cpp_fused_gelu_9(c_void_p(buf53.data_ptr()))
    buf54 = reinterpret_tensor(buf51, (128, 2560), (2560, 1), 0); del buf51  # reuse
    # Source Nodes: [hidden_states_20], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg34_1, reinterpret_tensor(buf53, (128, 10240), (10240, 1), 0), reinterpret_tensor(arg33_1, (10240, 2560), (1, 10240), 0), alpha=1, beta=1, out=buf54)
    del arg33_1
    del arg34_1
    buf55 = buf49; del buf49  # reuse
    buf56 = buf48; del buf48  # reuse
    cpp_fused_add_native_layer_norm_10(c_void_p(buf27.data_ptr()), c_void_p(buf47.data_ptr()), c_void_p(buf54.data_ptr()), c_void_p(buf55.data_ptr()), c_void_p(buf56.data_ptr()))
    buf62 = reinterpret_tensor(buf36, (128, 2560), (2560, 1), 0); del buf36  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_0_self_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg40_1, reinterpret_tensor(buf61, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg39_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf62)
    del arg39_1
    del arg40_1
    buf63 = reinterpret_tensor(buf35, (128, 2560), (2560, 1), 0); del buf35  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_0_self_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg42_1, reinterpret_tensor(buf61, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg41_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf63)
    del arg41_1
    del arg42_1
    buf64 = reinterpret_tensor(buf34, (1, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf34  # reuse
    buf65 = reinterpret_tensor(buf33, (1, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf33  # reuse
    cpp_fused_clone_11(c_void_p(buf62.data_ptr()), c_void_p(buf63.data_ptr()), c_void_p(buf64.data_ptr()), c_void_p(buf65.data_ptr()))
    buf66 = empty((32, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_weights_4], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf64, (32, 128, 80), (10240, 80, 1), 0), reinterpret_tensor(buf65, (32, 80, 128), (10240, 1, 80), 0), out=buf66)
    buf67 = empty_strided((32, 128, 1), (128, 1, 4096), device='cpu', dtype=torch.float32)
    buf68 = buf66; del buf66  # reuse
    buf69 = empty_strided((32, 128, 1), (128, 1, 4096), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_12(c_void_p(buf68.data_ptr()), c_void_p(buf67.data_ptr()), c_void_p(buf69.data_ptr()))
    buf70 = reinterpret_tensor(buf65, (128, 2560), (2560, 1), 0); del buf65  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_0_self_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg44_1, reinterpret_tensor(buf61, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg43_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf70)
    del arg43_1
    del arg44_1
    buf71 = buf68; del buf68  # reuse
    buf72 = reinterpret_tensor(buf61, (1, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf61  # reuse
    cpp_fused__softmax_clone_13(c_void_p(buf71.data_ptr()), c_void_p(buf69.data_ptr()), c_void_p(buf70.data_ptr()), c_void_p(buf72.data_ptr()))
    buf73 = reinterpret_tensor(buf70, (32, 128, 80), (10240, 80, 1), 0); del buf70  # reuse
    # Source Nodes: [attn_output_10, attn_weights_7], Original ATen: [aten._softmax, aten.bmm]
    extern_kernels.bmm(buf71, reinterpret_tensor(buf72, (32, 128, 80), (10240, 80, 1), 0), out=buf73)
    buf74 = reinterpret_tensor(buf72, (1, 128, 32, 80), (327680, 2560, 80, 1), 0); del buf72  # reuse
    cpp_fused_clone_14(c_void_p(buf73.data_ptr()), c_void_p(buf74.data_ptr()))
    buf75 = reinterpret_tensor(buf73, (128, 2560), (2560, 1), 0); del buf73  # reuse
    # Source Nodes: [hidden_states_28], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg46_1, reinterpret_tensor(buf74, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg45_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf75)
    del arg45_1
    del arg46_1
    buf76 = buf29; del buf29  # reuse
    buf77 = buf28; del buf28  # reuse
    buf79 = reinterpret_tensor(buf74, (1, 128, 2560), (327680, 2560, 1), 0); del buf74  # reuse
    cpp_fused_add_arange_embedding_mul_native_layer_norm_15(c_void_p(arg666_1.data_ptr()), c_void_p(arg2_1.data_ptr()), c_void_p(arg1_1.data_ptr()), c_void_p(buf75.data_ptr()), c_void_p(arg47_1.data_ptr()), c_void_p(arg48_1.data_ptr()), c_void_p(buf76.data_ptr()), c_void_p(buf77.data_ptr()), c_void_p(buf79.data_ptr()))
    del arg47_1
    del arg48_1
    del buf76
    del buf77
    buf80 = reinterpret_tensor(buf64, (128, 2560), (2560, 1), 0); del buf64  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_0_encoder_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg50_1, reinterpret_tensor(buf79, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg49_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf80)
    del arg49_1
    del arg50_1
    buf81 = buf79; del buf79  # reuse
    cpp_fused_add_native_layer_norm_16(c_void_p(buf27.data_ptr()), c_void_p(buf47.data_ptr()), c_void_p(buf54.data_ptr()), c_void_p(buf55.data_ptr()), c_void_p(buf56.data_ptr()), c_void_p(arg35_1.data_ptr()), c_void_p(arg36_1.data_ptr()), c_void_p(buf81.data_ptr()))
    del arg35_1
    del arg36_1
    buf82 = buf54; del buf54  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_0_encoder_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg52_1, reinterpret_tensor(buf81, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg51_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf82)
    del arg51_1
    del arg52_1
    buf83 = buf47; del buf47  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_0_encoder_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg54_1, reinterpret_tensor(buf81, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg53_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf83)
    del arg53_1
    del arg54_1
    buf84 = reinterpret_tensor(buf27, (1, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf27  # reuse
    buf85 = reinterpret_tensor(buf63, (1, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf63  # reuse
    buf86 = reinterpret_tensor(buf62, (1, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf62  # reuse
    cpp_fused_clone_17(c_void_p(buf80.data_ptr()), c_void_p(buf82.data_ptr()), c_void_p(buf83.data_ptr()), c_void_p(buf84.data_ptr()), c_void_p(buf85.data_ptr()), c_void_p(buf86.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf87 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf84, (1, 32, 128, 80), (0, 10240, 80, 1), 0), reinterpret_tensor(buf85, (1, 32, 128, 80), (0, 10240, 80, 1), 0), reinterpret_tensor(buf86, (1, 32, 128, 80), (0, 10240, 80, 1), 0), scale=1.0)
    buf88 = buf87[0]
    del buf87
    buf95 = reinterpret_tensor(buf88, (1, 128, 32, 80), (327680, 2560, 80, 1), 0); del buf88  # reuse
    cpp_fused_clone_18(c_void_p(buf95.data_ptr()))
    buf96 = reinterpret_tensor(buf86, (128, 2560), (2560, 1), 0); del buf86  # reuse
    # Source Nodes: [hidden_states_32], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg56_1, reinterpret_tensor(buf95, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg55_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf96)
    del arg55_1
    del arg56_1
    buf97 = reinterpret_tensor(buf96, (1, 128, 2560), (327680, 2560, 1), 0); del buf96  # reuse
    buf98 = buf56; del buf56  # reuse
    buf99 = buf55; del buf55  # reuse
    buf101 = reinterpret_tensor(buf95, (1, 128, 2560), (327680, 2560, 1), 0); del buf95  # reuse
    cpp_fused_add_arange_embedding_mul_native_layer_norm_19(c_void_p(buf97.data_ptr()), c_void_p(arg666_1.data_ptr()), c_void_p(arg2_1.data_ptr()), c_void_p(arg1_1.data_ptr()), c_void_p(buf75.data_ptr()), c_void_p(arg57_1.data_ptr()), c_void_p(arg58_1.data_ptr()), c_void_p(buf98.data_ptr()), c_void_p(buf99.data_ptr()), c_void_p(buf101.data_ptr()))
    del arg1_1
    del arg2_1
    del arg57_1
    del arg58_1
    del arg666_1
    buf102 = reinterpret_tensor(buf53, (128, 10240), (10240, 1), 0); del buf53  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_0_fc1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg60_1, reinterpret_tensor(buf101, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg59_1, (2560, 10240), (1, 2560), 0), alpha=1, beta=1, out=buf102)
    del arg59_1
    del arg60_1
    buf103 = reinterpret_tensor(buf102, (1, 128, 10240), (1310720, 10240, 1), 0); del buf102  # reuse
    cpp_fused_gelu_20(c_void_p(buf103.data_ptr()))
    buf104 = reinterpret_tensor(buf101, (128, 2560), (2560, 1), 0); del buf101  # reuse
    # Source Nodes: [hidden_states_38], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg62_1, reinterpret_tensor(buf103, (128, 10240), (10240, 1), 0), reinterpret_tensor(arg61_1, (10240, 2560), (1, 10240), 0), alpha=1, beta=1, out=buf104)
    del arg61_1
    del arg62_1
    buf105 = buf99; del buf99  # reuse
    buf106 = buf98; del buf98  # reuse
    buf108 = reinterpret_tensor(buf75, (1, 128, 2560), (327680, 2560, 1), 0); del buf75  # reuse
    cpp_fused_add_native_layer_norm_21(c_void_p(buf97.data_ptr()), c_void_p(buf104.data_ptr()), c_void_p(arg63_1.data_ptr()), c_void_p(arg64_1.data_ptr()), c_void_p(buf105.data_ptr()), c_void_p(buf106.data_ptr()), c_void_p(buf108.data_ptr()))
    del arg63_1
    del arg64_1
    buf109 = reinterpret_tensor(buf85, (128, 2560), (2560, 1), 0); del buf85  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_1_self_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg66_1, reinterpret_tensor(buf108, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg65_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf109)
    del arg65_1
    del arg66_1
    buf110 = reinterpret_tensor(buf84, (128, 2560), (2560, 1), 0); del buf84  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_1_self_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg68_1, reinterpret_tensor(buf108, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg67_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf110)
    del arg67_1
    del arg68_1
    buf111 = reinterpret_tensor(buf83, (1, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf83  # reuse
    buf112 = reinterpret_tensor(buf82, (1, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf82  # reuse
    cpp_fused_clone_22(c_void_p(buf109.data_ptr()), c_void_p(buf110.data_ptr()), c_void_p(buf111.data_ptr()), c_void_p(buf112.data_ptr()))
    buf113 = buf71; del buf71  # reuse
    # Source Nodes: [attn_weights_10], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf111, (32, 128, 80), (10240, 80, 1), 0), reinterpret_tensor(buf112, (32, 80, 128), (10240, 1, 80), 0), out=buf113)
    buf114 = buf69; del buf69  # reuse
    buf115 = buf113; del buf113  # reuse
    buf116 = buf67; del buf67  # reuse
    cpp_fused__softmax_23(c_void_p(buf115.data_ptr()), c_void_p(buf114.data_ptr()), c_void_p(buf116.data_ptr()))
    buf117 = reinterpret_tensor(buf112, (128, 2560), (2560, 1), 0); del buf112  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_1_self_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg70_1, reinterpret_tensor(buf108, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg69_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf117)
    del arg69_1
    del arg70_1
    buf118 = buf115; del buf115  # reuse
    buf119 = reinterpret_tensor(buf108, (1, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf108  # reuse
    cpp_fused__softmax_clone_24(c_void_p(buf118.data_ptr()), c_void_p(buf116.data_ptr()), c_void_p(buf117.data_ptr()), c_void_p(buf119.data_ptr()))
    buf120 = reinterpret_tensor(buf117, (32, 128, 80), (10240, 80, 1), 0); del buf117  # reuse
    # Source Nodes: [attn_output_20, attn_weights_13], Original ATen: [aten._softmax, aten.bmm]
    extern_kernels.bmm(buf118, reinterpret_tensor(buf119, (32, 128, 80), (10240, 80, 1), 0), out=buf120)
    buf121 = reinterpret_tensor(buf119, (1, 128, 32, 80), (327680, 2560, 80, 1), 0); del buf119  # reuse
    cpp_fused_clone_25(c_void_p(buf120.data_ptr()), c_void_p(buf121.data_ptr()))
    buf122 = reinterpret_tensor(buf120, (128, 2560), (2560, 1), 0); del buf120  # reuse
    # Source Nodes: [hidden_states_43], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg72_1, reinterpret_tensor(buf121, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg71_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf122)
    del arg71_1
    del arg72_1
    buf123 = buf106; del buf106  # reuse
    buf124 = buf105; del buf105  # reuse
    buf126 = reinterpret_tensor(buf121, (1, 128, 2560), (327680, 2560, 1), 0); del buf121  # reuse
    cpp_fused_add_native_layer_norm_26(c_void_p(buf97.data_ptr()), c_void_p(buf104.data_ptr()), c_void_p(buf122.data_ptr()), c_void_p(arg73_1.data_ptr()), c_void_p(arg74_1.data_ptr()), c_void_p(buf123.data_ptr()), c_void_p(buf124.data_ptr()), c_void_p(buf126.data_ptr()))
    del arg73_1
    del arg74_1
    buf127 = reinterpret_tensor(buf111, (128, 2560), (2560, 1), 0); del buf111  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_1_encoder_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg76_1, reinterpret_tensor(buf126, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg75_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf127)
    del arg75_1
    del arg76_1
    buf128 = reinterpret_tensor(buf126, (128, 2560), (2560, 1), 0); del buf126  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_1_encoder_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg78_1, reinterpret_tensor(buf81, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg77_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf128)
    del arg77_1
    del arg78_1
    buf129 = buf110; del buf110  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_1_encoder_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg80_1, reinterpret_tensor(buf81, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg79_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf129)
    del arg79_1
    del arg80_1
    buf130 = reinterpret_tensor(buf109, (1, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf109  # reuse
    buf131 = reinterpret_tensor(buf80, (1, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf80  # reuse
    buf132 = reinterpret_tensor(buf32, (1, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf32  # reuse
    cpp_fused_clone_27(c_void_p(buf127.data_ptr()), c_void_p(buf128.data_ptr()), c_void_p(buf129.data_ptr()), c_void_p(buf130.data_ptr()), c_void_p(buf131.data_ptr()), c_void_p(buf132.data_ptr()))
    del buf127
    del buf128
    # Source Nodes: [], Original ATen: []
    buf133 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf130, (1, 32, 128, 80), (0, 10240, 80, 1), 0), reinterpret_tensor(buf131, (1, 32, 128, 80), (0, 10240, 80, 1), 0), reinterpret_tensor(buf132, (1, 32, 128, 80), (0, 10240, 80, 1), 0), scale=1.0)
    buf134 = buf133[0]
    del buf133
    buf141 = reinterpret_tensor(buf134, (1, 128, 32, 80), (327680, 2560, 80, 1), 0); del buf134  # reuse
    cpp_fused_clone_28(c_void_p(buf141.data_ptr()))
    buf142 = reinterpret_tensor(buf132, (128, 2560), (2560, 1), 0); del buf132  # reuse
    # Source Nodes: [hidden_states_47], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg82_1, reinterpret_tensor(buf141, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg81_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf142)
    del arg81_1
    del arg82_1
    buf143 = buf124; del buf124  # reuse
    buf144 = buf123; del buf123  # reuse
    buf146 = reinterpret_tensor(buf141, (1, 128, 2560), (327680, 2560, 1), 0); del buf141  # reuse
    cpp_fused_add_native_layer_norm_29(c_void_p(buf97.data_ptr()), c_void_p(buf104.data_ptr()), c_void_p(buf122.data_ptr()), c_void_p(buf142.data_ptr()), c_void_p(arg83_1.data_ptr()), c_void_p(arg84_1.data_ptr()), c_void_p(buf143.data_ptr()), c_void_p(buf144.data_ptr()), c_void_p(buf146.data_ptr()))
    del arg83_1
    del arg84_1
    buf147 = reinterpret_tensor(buf103, (128, 10240), (10240, 1), 0); del buf103  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_1_fc1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg86_1, reinterpret_tensor(buf146, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg85_1, (2560, 10240), (1, 2560), 0), alpha=1, beta=1, out=buf147)
    del arg85_1
    del arg86_1
    buf148 = reinterpret_tensor(buf147, (1, 128, 10240), (1310720, 10240, 1), 0); del buf147  # reuse
    cpp_fused_gelu_30(c_void_p(buf148.data_ptr()))
    buf149 = reinterpret_tensor(buf146, (128, 2560), (2560, 1), 0); del buf146  # reuse
    # Source Nodes: [hidden_states_53], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg88_1, reinterpret_tensor(buf148, (128, 10240), (10240, 1), 0), reinterpret_tensor(arg87_1, (10240, 2560), (1, 10240), 0), alpha=1, beta=1, out=buf149)
    del arg87_1
    del arg88_1
    buf150 = reinterpret_tensor(buf149, (1, 128, 2560), (327680, 2560, 1), 0); del buf149  # reuse
    buf151 = buf144; del buf144  # reuse
    buf152 = buf143; del buf143  # reuse
    buf154 = reinterpret_tensor(buf131, (1, 128, 2560), (327680, 2560, 1), 0); del buf131  # reuse
    cpp_fused_add_native_layer_norm_31(c_void_p(buf150.data_ptr()), c_void_p(buf97.data_ptr()), c_void_p(buf104.data_ptr()), c_void_p(buf122.data_ptr()), c_void_p(buf142.data_ptr()), c_void_p(arg89_1.data_ptr()), c_void_p(arg90_1.data_ptr()), c_void_p(buf151.data_ptr()), c_void_p(buf152.data_ptr()), c_void_p(buf154.data_ptr()))
    del arg89_1
    del arg90_1
    buf155 = reinterpret_tensor(buf97, (128, 2560), (2560, 1), 0); del buf97  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_2_self_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg92_1, reinterpret_tensor(buf154, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg91_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf155)
    del arg91_1
    del arg92_1
    buf156 = buf142; del buf142  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_2_self_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg94_1, reinterpret_tensor(buf154, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg93_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf156)
    del arg93_1
    del arg94_1
    buf157 = reinterpret_tensor(buf122, (1, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf122  # reuse
    buf158 = reinterpret_tensor(buf104, (1, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf104  # reuse
    cpp_fused_clone_32(c_void_p(buf155.data_ptr()), c_void_p(buf156.data_ptr()), c_void_p(buf157.data_ptr()), c_void_p(buf158.data_ptr()))
    buf159 = buf118; del buf118  # reuse
    # Source Nodes: [attn_weights_16], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf157, (32, 128, 80), (10240, 80, 1), 0), reinterpret_tensor(buf158, (32, 80, 128), (10240, 1, 80), 0), out=buf159)
    buf160 = buf116; del buf116  # reuse
    buf161 = buf159; del buf159  # reuse
    buf162 = buf114; del buf114  # reuse
    cpp_fused__softmax_33(c_void_p(buf161.data_ptr()), c_void_p(buf160.data_ptr()), c_void_p(buf162.data_ptr()))
    buf163 = reinterpret_tensor(buf158, (128, 2560), (2560, 1), 0); del buf158  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_2_self_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg96_1, reinterpret_tensor(buf154, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg95_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf163)
    del arg95_1
    del arg96_1
    buf164 = buf161; del buf161  # reuse
    buf165 = reinterpret_tensor(buf154, (1, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf154  # reuse
    cpp_fused__softmax_clone_34(c_void_p(buf164.data_ptr()), c_void_p(buf162.data_ptr()), c_void_p(buf163.data_ptr()), c_void_p(buf165.data_ptr()))
    buf166 = reinterpret_tensor(buf163, (32, 128, 80), (10240, 80, 1), 0); del buf163  # reuse
    # Source Nodes: [attn_output_30, attn_weights_19], Original ATen: [aten._softmax, aten.bmm]
    extern_kernels.bmm(buf164, reinterpret_tensor(buf165, (32, 128, 80), (10240, 80, 1), 0), out=buf166)
    buf167 = reinterpret_tensor(buf165, (1, 128, 32, 80), (327680, 2560, 80, 1), 0); del buf165  # reuse
    cpp_fused_clone_35(c_void_p(buf166.data_ptr()), c_void_p(buf167.data_ptr()))
    buf168 = reinterpret_tensor(buf166, (128, 2560), (2560, 1), 0); del buf166  # reuse
    # Source Nodes: [hidden_states_58], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg98_1, reinterpret_tensor(buf167, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg97_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf168)
    del arg97_1
    del arg98_1
    buf169 = buf152; del buf152  # reuse
    buf170 = buf151; del buf151  # reuse
    buf172 = reinterpret_tensor(buf167, (1, 128, 2560), (327680, 2560, 1), 0); del buf167  # reuse
    cpp_fused_add_native_layer_norm_36(c_void_p(buf150.data_ptr()), c_void_p(buf168.data_ptr()), c_void_p(arg99_1.data_ptr()), c_void_p(arg100_1.data_ptr()), c_void_p(buf169.data_ptr()), c_void_p(buf170.data_ptr()), c_void_p(buf172.data_ptr()))
    del arg100_1
    del arg99_1
    buf173 = reinterpret_tensor(buf157, (128, 2560), (2560, 1), 0); del buf157  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_2_encoder_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg102_1, reinterpret_tensor(buf172, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg101_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf173)
    del arg101_1
    del arg102_1
    buf174 = reinterpret_tensor(buf172, (128, 2560), (2560, 1), 0); del buf172  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_2_encoder_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg104_1, reinterpret_tensor(buf81, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg103_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf174)
    del arg103_1
    del arg104_1
    buf175 = buf156; del buf156  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_2_encoder_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg106_1, reinterpret_tensor(buf81, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg105_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf175)
    del arg105_1
    del arg106_1
    buf176 = reinterpret_tensor(buf155, (1, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf155  # reuse
    buf177 = buf130; del buf130  # reuse
    buf178 = reinterpret_tensor(buf129, (1, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf129  # reuse
    cpp_fused_clone_37(c_void_p(buf173.data_ptr()), c_void_p(buf174.data_ptr()), c_void_p(buf175.data_ptr()), c_void_p(buf176.data_ptr()), c_void_p(buf177.data_ptr()), c_void_p(buf178.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf179 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf176, (1, 32, 128, 80), (0, 10240, 80, 1), 0), reinterpret_tensor(buf177, (1, 32, 128, 80), (0, 10240, 80, 1), 0), reinterpret_tensor(buf178, (1, 32, 128, 80), (0, 10240, 80, 1), 0), scale=1.0)
    buf180 = buf179[0]
    del buf179
    buf187 = reinterpret_tensor(buf180, (1, 128, 32, 80), (327680, 2560, 80, 1), 0); del buf180  # reuse
    cpp_fused_clone_38(c_void_p(buf187.data_ptr()))
    buf188 = reinterpret_tensor(buf178, (128, 2560), (2560, 1), 0); del buf178  # reuse
    # Source Nodes: [hidden_states_62], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg108_1, reinterpret_tensor(buf187, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg107_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf188)
    del arg107_1
    del arg108_1
    buf189 = buf170; del buf170  # reuse
    buf190 = buf169; del buf169  # reuse
    buf192 = reinterpret_tensor(buf187, (1, 128, 2560), (327680, 2560, 1), 0); del buf187  # reuse
    cpp_fused_add_native_layer_norm_39(c_void_p(buf150.data_ptr()), c_void_p(buf168.data_ptr()), c_void_p(buf188.data_ptr()), c_void_p(arg109_1.data_ptr()), c_void_p(arg110_1.data_ptr()), c_void_p(buf189.data_ptr()), c_void_p(buf190.data_ptr()), c_void_p(buf192.data_ptr()))
    del arg109_1
    del arg110_1
    buf193 = reinterpret_tensor(buf148, (128, 10240), (10240, 1), 0); del buf148  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_2_fc1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg112_1, reinterpret_tensor(buf192, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg111_1, (2560, 10240), (1, 2560), 0), alpha=1, beta=1, out=buf193)
    del arg111_1
    del arg112_1
    buf194 = reinterpret_tensor(buf193, (1, 128, 10240), (1310720, 10240, 1), 0); del buf193  # reuse
    cpp_fused_gelu_40(c_void_p(buf194.data_ptr()))
    buf195 = reinterpret_tensor(buf192, (128, 2560), (2560, 1), 0); del buf192  # reuse
    # Source Nodes: [hidden_states_68], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg114_1, reinterpret_tensor(buf194, (128, 10240), (10240, 1), 0), reinterpret_tensor(arg113_1, (10240, 2560), (1, 10240), 0), alpha=1, beta=1, out=buf195)
    del arg113_1
    del arg114_1
    buf196 = buf190; del buf190  # reuse
    buf197 = buf189; del buf189  # reuse
    buf199 = reinterpret_tensor(buf177, (1, 128, 2560), (327680, 2560, 1), 0); del buf177  # reuse
    cpp_fused_add_native_layer_norm_41(c_void_p(buf150.data_ptr()), c_void_p(buf168.data_ptr()), c_void_p(buf188.data_ptr()), c_void_p(buf195.data_ptr()), c_void_p(arg115_1.data_ptr()), c_void_p(arg116_1.data_ptr()), c_void_p(buf196.data_ptr()), c_void_p(buf197.data_ptr()), c_void_p(buf199.data_ptr()))
    del arg115_1
    del arg116_1
    buf200 = reinterpret_tensor(buf176, (128, 2560), (2560, 1), 0); del buf176  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_3_self_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg118_1, reinterpret_tensor(buf199, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg117_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf200)
    del arg117_1
    del arg118_1
    buf201 = buf175; del buf175  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_3_self_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg120_1, reinterpret_tensor(buf199, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg119_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf201)
    del arg119_1
    del arg120_1
    buf202 = reinterpret_tensor(buf174, (1, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf174  # reuse
    buf203 = reinterpret_tensor(buf173, (1, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf173  # reuse
    cpp_fused_clone_42(c_void_p(buf200.data_ptr()), c_void_p(buf201.data_ptr()), c_void_p(buf202.data_ptr()), c_void_p(buf203.data_ptr()))
    buf204 = buf164; del buf164  # reuse
    # Source Nodes: [attn_weights_22], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf202, (32, 128, 80), (10240, 80, 1), 0), reinterpret_tensor(buf203, (32, 80, 128), (10240, 1, 80), 0), out=buf204)
    buf205 = buf162; del buf162  # reuse
    buf206 = buf204; del buf204  # reuse
    buf207 = buf160; del buf160  # reuse
    cpp_fused__softmax_43(c_void_p(buf206.data_ptr()), c_void_p(buf205.data_ptr()), c_void_p(buf207.data_ptr()))
    buf208 = reinterpret_tensor(buf203, (128, 2560), (2560, 1), 0); del buf203  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_3_self_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg122_1, reinterpret_tensor(buf199, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg121_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf208)
    del arg121_1
    del arg122_1
    buf209 = buf206; del buf206  # reuse
    buf210 = reinterpret_tensor(buf199, (1, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf199  # reuse
    cpp_fused__softmax_clone_44(c_void_p(buf209.data_ptr()), c_void_p(buf207.data_ptr()), c_void_p(buf208.data_ptr()), c_void_p(buf210.data_ptr()))
    buf211 = reinterpret_tensor(buf208, (32, 128, 80), (10240, 80, 1), 0); del buf208  # reuse
    # Source Nodes: [attn_output_40, attn_weights_25], Original ATen: [aten._softmax, aten.bmm]
    extern_kernels.bmm(buf209, reinterpret_tensor(buf210, (32, 128, 80), (10240, 80, 1), 0), out=buf211)
    buf212 = reinterpret_tensor(buf210, (1, 128, 32, 80), (327680, 2560, 80, 1), 0); del buf210  # reuse
    cpp_fused_clone_45(c_void_p(buf211.data_ptr()), c_void_p(buf212.data_ptr()))
    buf213 = reinterpret_tensor(buf211, (128, 2560), (2560, 1), 0); del buf211  # reuse
    # Source Nodes: [hidden_states_73], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg124_1, reinterpret_tensor(buf212, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg123_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf213)
    del arg123_1
    del arg124_1
    buf214 = reinterpret_tensor(buf213, (1, 128, 2560), (327680, 2560, 1), 0); del buf213  # reuse
    buf215 = buf197; del buf197  # reuse
    buf216 = buf196; del buf196  # reuse
    buf218 = reinterpret_tensor(buf212, (1, 128, 2560), (327680, 2560, 1), 0); del buf212  # reuse
    cpp_fused_add_native_layer_norm_46(c_void_p(buf214.data_ptr()), c_void_p(buf150.data_ptr()), c_void_p(buf168.data_ptr()), c_void_p(buf188.data_ptr()), c_void_p(buf195.data_ptr()), c_void_p(arg125_1.data_ptr()), c_void_p(arg126_1.data_ptr()), c_void_p(buf215.data_ptr()), c_void_p(buf216.data_ptr()), c_void_p(buf218.data_ptr()))
    del arg125_1
    del arg126_1
    buf219 = buf195; del buf195  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_3_encoder_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg128_1, reinterpret_tensor(buf218, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg127_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf219)
    del arg127_1
    del arg128_1
    buf220 = reinterpret_tensor(buf218, (128, 2560), (2560, 1), 0); del buf218  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_3_encoder_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg130_1, reinterpret_tensor(buf81, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg129_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf220)
    del arg129_1
    del arg130_1
    buf221 = buf188; del buf188  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_3_encoder_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg132_1, reinterpret_tensor(buf81, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg131_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf221)
    del arg131_1
    del arg132_1
    buf222 = reinterpret_tensor(buf168, (1, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf168  # reuse
    buf223 = reinterpret_tensor(buf150, (1, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf150  # reuse
    buf224 = buf202; del buf202  # reuse
    cpp_fused_clone_47(c_void_p(buf219.data_ptr()), c_void_p(buf220.data_ptr()), c_void_p(buf221.data_ptr()), c_void_p(buf222.data_ptr()), c_void_p(buf223.data_ptr()), c_void_p(buf224.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf225 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf222, (1, 32, 128, 80), (0, 10240, 80, 1), 0), reinterpret_tensor(buf223, (1, 32, 128, 80), (0, 10240, 80, 1), 0), reinterpret_tensor(buf224, (1, 32, 128, 80), (0, 10240, 80, 1), 0), scale=1.0)
    buf226 = buf225[0]
    del buf225
    buf233 = reinterpret_tensor(buf226, (1, 128, 32, 80), (327680, 2560, 80, 1), 0); del buf226  # reuse
    cpp_fused_clone_48(c_void_p(buf233.data_ptr()))
    buf234 = reinterpret_tensor(buf224, (128, 2560), (2560, 1), 0); del buf224  # reuse
    # Source Nodes: [hidden_states_77], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg134_1, reinterpret_tensor(buf233, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg133_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf234)
    del arg133_1
    del arg134_1
    buf235 = buf216; del buf216  # reuse
    buf236 = buf215; del buf215  # reuse
    buf238 = reinterpret_tensor(buf233, (1, 128, 2560), (327680, 2560, 1), 0); del buf233  # reuse
    cpp_fused_add_native_layer_norm_49(c_void_p(buf214.data_ptr()), c_void_p(buf234.data_ptr()), c_void_p(arg135_1.data_ptr()), c_void_p(arg136_1.data_ptr()), c_void_p(buf235.data_ptr()), c_void_p(buf236.data_ptr()), c_void_p(buf238.data_ptr()))
    del arg135_1
    del arg136_1
    buf239 = reinterpret_tensor(buf194, (128, 10240), (10240, 1), 0); del buf194  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_3_fc1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg138_1, reinterpret_tensor(buf238, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg137_1, (2560, 10240), (1, 2560), 0), alpha=1, beta=1, out=buf239)
    del arg137_1
    del arg138_1
    buf240 = reinterpret_tensor(buf239, (1, 128, 10240), (1310720, 10240, 1), 0); del buf239  # reuse
    cpp_fused_gelu_50(c_void_p(buf240.data_ptr()))
    buf241 = reinterpret_tensor(buf238, (128, 2560), (2560, 1), 0); del buf238  # reuse
    # Source Nodes: [hidden_states_83], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg140_1, reinterpret_tensor(buf240, (128, 10240), (10240, 1), 0), reinterpret_tensor(arg139_1, (10240, 2560), (1, 10240), 0), alpha=1, beta=1, out=buf241)
    del arg139_1
    del arg140_1
    buf242 = buf236; del buf236  # reuse
    buf243 = buf235; del buf235  # reuse
    buf245 = reinterpret_tensor(buf223, (1, 128, 2560), (327680, 2560, 1), 0); del buf223  # reuse
    cpp_fused_add_native_layer_norm_51(c_void_p(buf214.data_ptr()), c_void_p(buf234.data_ptr()), c_void_p(buf241.data_ptr()), c_void_p(arg141_1.data_ptr()), c_void_p(arg142_1.data_ptr()), c_void_p(buf242.data_ptr()), c_void_p(buf243.data_ptr()), c_void_p(buf245.data_ptr()))
    del arg141_1
    del arg142_1
    buf246 = reinterpret_tensor(buf222, (128, 2560), (2560, 1), 0); del buf222  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_4_self_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg144_1, reinterpret_tensor(buf245, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg143_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf246)
    del arg143_1
    del arg144_1
    buf247 = buf221; del buf221  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_4_self_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg146_1, reinterpret_tensor(buf245, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg145_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf247)
    del arg145_1
    del arg146_1
    buf248 = reinterpret_tensor(buf220, (1, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf220  # reuse
    buf249 = reinterpret_tensor(buf219, (1, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf219  # reuse
    cpp_fused_clone_52(c_void_p(buf246.data_ptr()), c_void_p(buf247.data_ptr()), c_void_p(buf248.data_ptr()), c_void_p(buf249.data_ptr()))
    buf250 = buf209; del buf209  # reuse
    # Source Nodes: [attn_weights_28], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf248, (32, 128, 80), (10240, 80, 1), 0), reinterpret_tensor(buf249, (32, 80, 128), (10240, 1, 80), 0), out=buf250)
    buf251 = buf207; del buf207  # reuse
    buf252 = buf250; del buf250  # reuse
    buf253 = buf205; del buf205  # reuse
    cpp_fused__softmax_53(c_void_p(buf252.data_ptr()), c_void_p(buf251.data_ptr()), c_void_p(buf253.data_ptr()))
    buf254 = reinterpret_tensor(buf249, (128, 2560), (2560, 1), 0); del buf249  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_4_self_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg148_1, reinterpret_tensor(buf245, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg147_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf254)
    del arg147_1
    del arg148_1
    buf255 = buf252; del buf252  # reuse
    buf256 = reinterpret_tensor(buf245, (1, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf245  # reuse
    cpp_fused__softmax_clone_54(c_void_p(buf255.data_ptr()), c_void_p(buf253.data_ptr()), c_void_p(buf254.data_ptr()), c_void_p(buf256.data_ptr()))
    buf257 = reinterpret_tensor(buf254, (32, 128, 80), (10240, 80, 1), 0); del buf254  # reuse
    # Source Nodes: [attn_output_50, attn_weights_31], Original ATen: [aten._softmax, aten.bmm]
    extern_kernels.bmm(buf255, reinterpret_tensor(buf256, (32, 128, 80), (10240, 80, 1), 0), out=buf257)
    buf258 = reinterpret_tensor(buf256, (1, 128, 32, 80), (327680, 2560, 80, 1), 0); del buf256  # reuse
    cpp_fused_clone_55(c_void_p(buf257.data_ptr()), c_void_p(buf258.data_ptr()))
    buf259 = reinterpret_tensor(buf257, (128, 2560), (2560, 1), 0); del buf257  # reuse
    # Source Nodes: [hidden_states_88], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg150_1, reinterpret_tensor(buf258, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg149_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf259)
    del arg149_1
    del arg150_1
    buf260 = buf243; del buf243  # reuse
    buf261 = buf242; del buf242  # reuse
    buf263 = reinterpret_tensor(buf258, (1, 128, 2560), (327680, 2560, 1), 0); del buf258  # reuse
    cpp_fused_add_native_layer_norm_56(c_void_p(buf214.data_ptr()), c_void_p(buf234.data_ptr()), c_void_p(buf241.data_ptr()), c_void_p(buf259.data_ptr()), c_void_p(arg151_1.data_ptr()), c_void_p(arg152_1.data_ptr()), c_void_p(buf260.data_ptr()), c_void_p(buf261.data_ptr()), c_void_p(buf263.data_ptr()))
    del arg151_1
    del arg152_1
    buf264 = reinterpret_tensor(buf248, (128, 2560), (2560, 1), 0); del buf248  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_4_encoder_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg154_1, reinterpret_tensor(buf263, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg153_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf264)
    del arg153_1
    del arg154_1
    buf265 = reinterpret_tensor(buf263, (128, 2560), (2560, 1), 0); del buf263  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_4_encoder_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg156_1, reinterpret_tensor(buf81, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg155_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf265)
    del arg155_1
    del arg156_1
    buf266 = buf247; del buf247  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_4_encoder_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg158_1, reinterpret_tensor(buf81, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg157_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf266)
    del arg157_1
    del arg158_1
    buf267 = reinterpret_tensor(buf246, (1, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf246  # reuse
    buf268 = reinterpret_tensor(buf201, (1, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf201  # reuse
    buf269 = reinterpret_tensor(buf200, (1, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf200  # reuse
    cpp_fused_clone_57(c_void_p(buf264.data_ptr()), c_void_p(buf265.data_ptr()), c_void_p(buf266.data_ptr()), c_void_p(buf267.data_ptr()), c_void_p(buf268.data_ptr()), c_void_p(buf269.data_ptr()))
    del buf264
    del buf265
    # Source Nodes: [], Original ATen: []
    buf270 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf267, (1, 32, 128, 80), (0, 10240, 80, 1), 0), reinterpret_tensor(buf268, (1, 32, 128, 80), (0, 10240, 80, 1), 0), reinterpret_tensor(buf269, (1, 32, 128, 80), (0, 10240, 80, 1), 0), scale=1.0)
    buf271 = buf270[0]
    del buf270
    buf278 = reinterpret_tensor(buf271, (1, 128, 32, 80), (327680, 2560, 80, 1), 0); del buf271  # reuse
    cpp_fused_clone_58(c_void_p(buf278.data_ptr()))
    buf279 = reinterpret_tensor(buf269, (128, 2560), (2560, 1), 0); del buf269  # reuse
    # Source Nodes: [hidden_states_92], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg160_1, reinterpret_tensor(buf278, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg159_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf279)
    del arg159_1
    del arg160_1
    buf280 = reinterpret_tensor(buf279, (1, 128, 2560), (327680, 2560, 1), 0); del buf279  # reuse
    buf281 = buf261; del buf261  # reuse
    buf282 = buf260; del buf260  # reuse
    buf284 = reinterpret_tensor(buf278, (1, 128, 2560), (327680, 2560, 1), 0); del buf278  # reuse
    cpp_fused_add_native_layer_norm_59(c_void_p(buf280.data_ptr()), c_void_p(buf214.data_ptr()), c_void_p(buf234.data_ptr()), c_void_p(buf241.data_ptr()), c_void_p(buf259.data_ptr()), c_void_p(arg161_1.data_ptr()), c_void_p(arg162_1.data_ptr()), c_void_p(buf281.data_ptr()), c_void_p(buf282.data_ptr()), c_void_p(buf284.data_ptr()))
    del arg161_1
    del arg162_1
    buf285 = reinterpret_tensor(buf240, (128, 10240), (10240, 1), 0); del buf240  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_4_fc1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg164_1, reinterpret_tensor(buf284, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg163_1, (2560, 10240), (1, 2560), 0), alpha=1, beta=1, out=buf285)
    del arg163_1
    del arg164_1
    buf286 = reinterpret_tensor(buf285, (1, 128, 10240), (1310720, 10240, 1), 0); del buf285  # reuse
    cpp_fused_gelu_60(c_void_p(buf286.data_ptr()))
    buf287 = reinterpret_tensor(buf284, (128, 2560), (2560, 1), 0); del buf284  # reuse
    # Source Nodes: [hidden_states_98], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg166_1, reinterpret_tensor(buf286, (128, 10240), (10240, 1), 0), reinterpret_tensor(arg165_1, (10240, 2560), (1, 10240), 0), alpha=1, beta=1, out=buf287)
    del arg165_1
    del arg166_1
    buf288 = buf282; del buf282  # reuse
    buf289 = buf281; del buf281  # reuse
    buf291 = reinterpret_tensor(buf259, (1, 128, 2560), (327680, 2560, 1), 0); del buf259  # reuse
    cpp_fused_add_native_layer_norm_61(c_void_p(buf280.data_ptr()), c_void_p(buf287.data_ptr()), c_void_p(arg167_1.data_ptr()), c_void_p(arg168_1.data_ptr()), c_void_p(buf288.data_ptr()), c_void_p(buf289.data_ptr()), c_void_p(buf291.data_ptr()))
    del arg167_1
    del arg168_1
    buf292 = buf241; del buf241  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_5_self_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg170_1, reinterpret_tensor(buf291, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg169_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf292)
    del arg169_1
    del arg170_1
    buf293 = buf234; del buf234  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_5_self_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg172_1, reinterpret_tensor(buf291, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg171_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf293)
    del arg171_1
    del arg172_1
    buf294 = reinterpret_tensor(buf214, (1, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf214  # reuse
    buf295 = buf268; del buf268  # reuse
    cpp_fused_clone_62(c_void_p(buf292.data_ptr()), c_void_p(buf293.data_ptr()), c_void_p(buf294.data_ptr()), c_void_p(buf295.data_ptr()))
    buf296 = buf255; del buf255  # reuse
    # Source Nodes: [attn_weights_34], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf294, (32, 128, 80), (10240, 80, 1), 0), reinterpret_tensor(buf295, (32, 80, 128), (10240, 1, 80), 0), out=buf296)
    buf297 = buf253; del buf253  # reuse
    buf298 = buf296; del buf296  # reuse
    buf299 = buf251; del buf251  # reuse
    cpp_fused__softmax_63(c_void_p(buf298.data_ptr()), c_void_p(buf297.data_ptr()), c_void_p(buf299.data_ptr()))
    buf300 = reinterpret_tensor(buf295, (128, 2560), (2560, 1), 0); del buf295  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_5_self_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg174_1, reinterpret_tensor(buf291, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg173_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf300)
    del arg173_1
    del arg174_1
    buf301 = buf298; del buf298  # reuse
    buf302 = reinterpret_tensor(buf291, (1, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf291  # reuse
    cpp_fused__softmax_clone_64(c_void_p(buf301.data_ptr()), c_void_p(buf299.data_ptr()), c_void_p(buf300.data_ptr()), c_void_p(buf302.data_ptr()))
    buf303 = reinterpret_tensor(buf300, (32, 128, 80), (10240, 80, 1), 0); del buf300  # reuse
    # Source Nodes: [attn_output_60, attn_weights_37], Original ATen: [aten._softmax, aten.bmm]
    extern_kernels.bmm(buf301, reinterpret_tensor(buf302, (32, 128, 80), (10240, 80, 1), 0), out=buf303)
    buf304 = reinterpret_tensor(buf302, (1, 128, 32, 80), (327680, 2560, 80, 1), 0); del buf302  # reuse
    cpp_fused_clone_65(c_void_p(buf303.data_ptr()), c_void_p(buf304.data_ptr()))
    buf305 = reinterpret_tensor(buf303, (128, 2560), (2560, 1), 0); del buf303  # reuse
    # Source Nodes: [hidden_states_103], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg176_1, reinterpret_tensor(buf304, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg175_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf305)
    del arg175_1
    del arg176_1
    buf306 = buf289; del buf289  # reuse
    buf307 = buf288; del buf288  # reuse
    buf309 = reinterpret_tensor(buf304, (1, 128, 2560), (327680, 2560, 1), 0); del buf304  # reuse
    cpp_fused_add_native_layer_norm_66(c_void_p(buf280.data_ptr()), c_void_p(buf287.data_ptr()), c_void_p(buf305.data_ptr()), c_void_p(arg177_1.data_ptr()), c_void_p(arg178_1.data_ptr()), c_void_p(buf306.data_ptr()), c_void_p(buf307.data_ptr()), c_void_p(buf309.data_ptr()))
    del arg177_1
    del arg178_1
    buf310 = reinterpret_tensor(buf294, (128, 2560), (2560, 1), 0); del buf294  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_5_encoder_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg180_1, reinterpret_tensor(buf309, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg179_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf310)
    del arg179_1
    del arg180_1
    buf311 = reinterpret_tensor(buf309, (128, 2560), (2560, 1), 0); del buf309  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_5_encoder_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg182_1, reinterpret_tensor(buf81, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg181_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf311)
    del arg181_1
    del arg182_1
    buf312 = buf293; del buf293  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_5_encoder_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg184_1, reinterpret_tensor(buf81, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg183_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf312)
    del arg183_1
    del arg184_1
    buf313 = reinterpret_tensor(buf292, (1, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf292  # reuse
    buf314 = buf267; del buf267  # reuse
    buf315 = reinterpret_tensor(buf266, (1, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf266  # reuse
    cpp_fused_clone_67(c_void_p(buf310.data_ptr()), c_void_p(buf311.data_ptr()), c_void_p(buf312.data_ptr()), c_void_p(buf313.data_ptr()), c_void_p(buf314.data_ptr()), c_void_p(buf315.data_ptr()))
    del buf310
    del buf311
    # Source Nodes: [], Original ATen: []
    buf316 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf313, (1, 32, 128, 80), (0, 10240, 80, 1), 0), reinterpret_tensor(buf314, (1, 32, 128, 80), (0, 10240, 80, 1), 0), reinterpret_tensor(buf315, (1, 32, 128, 80), (0, 10240, 80, 1), 0), scale=1.0)
    buf317 = buf316[0]
    del buf316
    buf324 = reinterpret_tensor(buf317, (1, 128, 32, 80), (327680, 2560, 80, 1), 0); del buf317  # reuse
    cpp_fused_clone_68(c_void_p(buf324.data_ptr()))
    buf325 = reinterpret_tensor(buf315, (128, 2560), (2560, 1), 0); del buf315  # reuse
    # Source Nodes: [hidden_states_107], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg186_1, reinterpret_tensor(buf324, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg185_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf325)
    del arg185_1
    del arg186_1
    buf326 = buf307; del buf307  # reuse
    buf327 = buf306; del buf306  # reuse
    buf329 = reinterpret_tensor(buf324, (1, 128, 2560), (327680, 2560, 1), 0); del buf324  # reuse
    cpp_fused_add_native_layer_norm_69(c_void_p(buf280.data_ptr()), c_void_p(buf287.data_ptr()), c_void_p(buf305.data_ptr()), c_void_p(buf325.data_ptr()), c_void_p(arg187_1.data_ptr()), c_void_p(arg188_1.data_ptr()), c_void_p(buf326.data_ptr()), c_void_p(buf327.data_ptr()), c_void_p(buf329.data_ptr()))
    del arg187_1
    del arg188_1
    buf330 = reinterpret_tensor(buf286, (128, 10240), (10240, 1), 0); del buf286  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_5_fc1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg190_1, reinterpret_tensor(buf329, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg189_1, (2560, 10240), (1, 2560), 0), alpha=1, beta=1, out=buf330)
    del arg189_1
    del arg190_1
    buf331 = reinterpret_tensor(buf330, (1, 128, 10240), (1310720, 10240, 1), 0); del buf330  # reuse
    cpp_fused_gelu_70(c_void_p(buf331.data_ptr()))
    buf332 = reinterpret_tensor(buf329, (128, 2560), (2560, 1), 0); del buf329  # reuse
    # Source Nodes: [hidden_states_113], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg192_1, reinterpret_tensor(buf331, (128, 10240), (10240, 1), 0), reinterpret_tensor(arg191_1, (10240, 2560), (1, 10240), 0), alpha=1, beta=1, out=buf332)
    del arg191_1
    del arg192_1
    buf333 = reinterpret_tensor(buf332, (1, 128, 2560), (327680, 2560, 1), 0); del buf332  # reuse
    buf334 = buf327; del buf327  # reuse
    buf335 = buf326; del buf326  # reuse
    buf337 = reinterpret_tensor(buf314, (1, 128, 2560), (327680, 2560, 1), 0); del buf314  # reuse
    cpp_fused_add_native_layer_norm_71(c_void_p(buf333.data_ptr()), c_void_p(buf280.data_ptr()), c_void_p(buf287.data_ptr()), c_void_p(buf305.data_ptr()), c_void_p(buf325.data_ptr()), c_void_p(arg193_1.data_ptr()), c_void_p(arg194_1.data_ptr()), c_void_p(buf334.data_ptr()), c_void_p(buf335.data_ptr()), c_void_p(buf337.data_ptr()))
    del arg193_1
    del arg194_1
    buf338 = buf325; del buf325  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_6_self_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg196_1, reinterpret_tensor(buf337, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg195_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf338)
    del arg195_1
    del arg196_1
    buf339 = buf305; del buf305  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_6_self_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg198_1, reinterpret_tensor(buf337, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg197_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf339)
    del arg197_1
    del arg198_1
    buf340 = reinterpret_tensor(buf287, (1, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf287  # reuse
    buf341 = reinterpret_tensor(buf280, (1, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf280  # reuse
    cpp_fused_clone_72(c_void_p(buf338.data_ptr()), c_void_p(buf339.data_ptr()), c_void_p(buf340.data_ptr()), c_void_p(buf341.data_ptr()))
    buf342 = buf301; del buf301  # reuse
    # Source Nodes: [attn_weights_40], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf340, (32, 128, 80), (10240, 80, 1), 0), reinterpret_tensor(buf341, (32, 80, 128), (10240, 1, 80), 0), out=buf342)
    buf343 = buf299; del buf299  # reuse
    buf344 = buf342; del buf342  # reuse
    buf345 = buf297; del buf297  # reuse
    cpp_fused__softmax_73(c_void_p(buf344.data_ptr()), c_void_p(buf343.data_ptr()), c_void_p(buf345.data_ptr()))
    buf346 = reinterpret_tensor(buf341, (128, 2560), (2560, 1), 0); del buf341  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_6_self_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg200_1, reinterpret_tensor(buf337, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg199_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf346)
    del arg199_1
    del arg200_1
    buf347 = buf344; del buf344  # reuse
    buf348 = reinterpret_tensor(buf337, (1, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf337  # reuse
    cpp_fused__softmax_clone_74(c_void_p(buf347.data_ptr()), c_void_p(buf345.data_ptr()), c_void_p(buf346.data_ptr()), c_void_p(buf348.data_ptr()))
    buf349 = reinterpret_tensor(buf346, (32, 128, 80), (10240, 80, 1), 0); del buf346  # reuse
    # Source Nodes: [attn_output_70, attn_weights_43], Original ATen: [aten._softmax, aten.bmm]
    extern_kernels.bmm(buf347, reinterpret_tensor(buf348, (32, 128, 80), (10240, 80, 1), 0), out=buf349)
    buf350 = reinterpret_tensor(buf348, (1, 128, 32, 80), (327680, 2560, 80, 1), 0); del buf348  # reuse
    cpp_fused_clone_75(c_void_p(buf349.data_ptr()), c_void_p(buf350.data_ptr()))
    buf351 = reinterpret_tensor(buf349, (128, 2560), (2560, 1), 0); del buf349  # reuse
    # Source Nodes: [hidden_states_118], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg202_1, reinterpret_tensor(buf350, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg201_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf351)
    del arg201_1
    del arg202_1
    buf352 = buf335; del buf335  # reuse
    buf353 = buf334; del buf334  # reuse
    buf355 = reinterpret_tensor(buf350, (1, 128, 2560), (327680, 2560, 1), 0); del buf350  # reuse
    cpp_fused_add_native_layer_norm_76(c_void_p(buf333.data_ptr()), c_void_p(buf351.data_ptr()), c_void_p(arg203_1.data_ptr()), c_void_p(arg204_1.data_ptr()), c_void_p(buf352.data_ptr()), c_void_p(buf353.data_ptr()), c_void_p(buf355.data_ptr()))
    del arg203_1
    del arg204_1
    buf356 = reinterpret_tensor(buf340, (128, 2560), (2560, 1), 0); del buf340  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_6_encoder_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg206_1, reinterpret_tensor(buf355, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg205_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf356)
    del arg205_1
    del arg206_1
    buf357 = reinterpret_tensor(buf355, (128, 2560), (2560, 1), 0); del buf355  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_6_encoder_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg208_1, reinterpret_tensor(buf81, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg207_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf357)
    del arg207_1
    del arg208_1
    buf358 = buf339; del buf339  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_6_encoder_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg210_1, reinterpret_tensor(buf81, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg209_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf358)
    del arg209_1
    del arg210_1
    buf359 = reinterpret_tensor(buf338, (1, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf338  # reuse
    buf360 = buf313; del buf313  # reuse
    buf361 = reinterpret_tensor(buf312, (1, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf312  # reuse
    cpp_fused_clone_77(c_void_p(buf356.data_ptr()), c_void_p(buf357.data_ptr()), c_void_p(buf358.data_ptr()), c_void_p(buf359.data_ptr()), c_void_p(buf360.data_ptr()), c_void_p(buf361.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf362 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf359, (1, 32, 128, 80), (0, 10240, 80, 1), 0), reinterpret_tensor(buf360, (1, 32, 128, 80), (0, 10240, 80, 1), 0), reinterpret_tensor(buf361, (1, 32, 128, 80), (0, 10240, 80, 1), 0), scale=1.0)
    buf363 = buf362[0]
    del buf362
    buf370 = reinterpret_tensor(buf363, (1, 128, 32, 80), (327680, 2560, 80, 1), 0); del buf363  # reuse
    cpp_fused_clone_78(c_void_p(buf370.data_ptr()))
    buf371 = reinterpret_tensor(buf361, (128, 2560), (2560, 1), 0); del buf361  # reuse
    # Source Nodes: [hidden_states_122], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg212_1, reinterpret_tensor(buf370, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg211_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf371)
    del arg211_1
    del arg212_1
    buf372 = buf353; del buf353  # reuse
    buf373 = buf352; del buf352  # reuse
    buf375 = reinterpret_tensor(buf370, (1, 128, 2560), (327680, 2560, 1), 0); del buf370  # reuse
    cpp_fused_add_native_layer_norm_79(c_void_p(buf333.data_ptr()), c_void_p(buf351.data_ptr()), c_void_p(buf371.data_ptr()), c_void_p(arg213_1.data_ptr()), c_void_p(arg214_1.data_ptr()), c_void_p(buf372.data_ptr()), c_void_p(buf373.data_ptr()), c_void_p(buf375.data_ptr()))
    del arg213_1
    del arg214_1
    buf376 = reinterpret_tensor(buf331, (128, 10240), (10240, 1), 0); del buf331  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_6_fc1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg216_1, reinterpret_tensor(buf375, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg215_1, (2560, 10240), (1, 2560), 0), alpha=1, beta=1, out=buf376)
    del arg215_1
    del arg216_1
    buf377 = reinterpret_tensor(buf376, (1, 128, 10240), (1310720, 10240, 1), 0); del buf376  # reuse
    cpp_fused_gelu_80(c_void_p(buf377.data_ptr()))
    buf378 = reinterpret_tensor(buf375, (128, 2560), (2560, 1), 0); del buf375  # reuse
    # Source Nodes: [hidden_states_128], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg218_1, reinterpret_tensor(buf377, (128, 10240), (10240, 1), 0), reinterpret_tensor(arg217_1, (10240, 2560), (1, 10240), 0), alpha=1, beta=1, out=buf378)
    del arg217_1
    del arg218_1
    buf379 = buf373; del buf373  # reuse
    buf380 = buf372; del buf372  # reuse
    buf382 = reinterpret_tensor(buf360, (1, 128, 2560), (327680, 2560, 1), 0); del buf360  # reuse
    cpp_fused_add_native_layer_norm_81(c_void_p(buf333.data_ptr()), c_void_p(buf351.data_ptr()), c_void_p(buf371.data_ptr()), c_void_p(buf378.data_ptr()), c_void_p(arg219_1.data_ptr()), c_void_p(arg220_1.data_ptr()), c_void_p(buf379.data_ptr()), c_void_p(buf380.data_ptr()), c_void_p(buf382.data_ptr()))
    del arg219_1
    del arg220_1
    buf383 = reinterpret_tensor(buf359, (128, 2560), (2560, 1), 0); del buf359  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_7_self_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg222_1, reinterpret_tensor(buf382, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg221_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf383)
    del arg221_1
    del arg222_1
    buf384 = buf358; del buf358  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_7_self_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg224_1, reinterpret_tensor(buf382, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg223_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf384)
    del arg223_1
    del arg224_1
    buf385 = reinterpret_tensor(buf357, (1, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf357  # reuse
    buf386 = reinterpret_tensor(buf356, (1, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf356  # reuse
    cpp_fused_clone_82(c_void_p(buf383.data_ptr()), c_void_p(buf384.data_ptr()), c_void_p(buf385.data_ptr()), c_void_p(buf386.data_ptr()))
    buf387 = buf347; del buf347  # reuse
    # Source Nodes: [attn_weights_46], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf385, (32, 128, 80), (10240, 80, 1), 0), reinterpret_tensor(buf386, (32, 80, 128), (10240, 1, 80), 0), out=buf387)
    buf388 = buf345; del buf345  # reuse
    buf389 = buf387; del buf387  # reuse
    buf390 = buf343; del buf343  # reuse
    cpp_fused__softmax_83(c_void_p(buf389.data_ptr()), c_void_p(buf388.data_ptr()), c_void_p(buf390.data_ptr()))
    buf391 = reinterpret_tensor(buf386, (128, 2560), (2560, 1), 0); del buf386  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_7_self_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg226_1, reinterpret_tensor(buf382, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg225_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf391)
    del arg225_1
    del arg226_1
    buf392 = buf389; del buf389  # reuse
    buf393 = reinterpret_tensor(buf382, (1, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf382  # reuse
    cpp_fused__softmax_clone_84(c_void_p(buf392.data_ptr()), c_void_p(buf390.data_ptr()), c_void_p(buf391.data_ptr()), c_void_p(buf393.data_ptr()))
    buf394 = reinterpret_tensor(buf391, (32, 128, 80), (10240, 80, 1), 0); del buf391  # reuse
    # Source Nodes: [attn_output_80, attn_weights_49], Original ATen: [aten._softmax, aten.bmm]
    extern_kernels.bmm(buf392, reinterpret_tensor(buf393, (32, 128, 80), (10240, 80, 1), 0), out=buf394)
    buf395 = reinterpret_tensor(buf393, (1, 128, 32, 80), (327680, 2560, 80, 1), 0); del buf393  # reuse
    cpp_fused_clone_85(c_void_p(buf394.data_ptr()), c_void_p(buf395.data_ptr()))
    buf396 = reinterpret_tensor(buf394, (128, 2560), (2560, 1), 0); del buf394  # reuse
    # Source Nodes: [hidden_states_133], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg228_1, reinterpret_tensor(buf395, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg227_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf396)
    del arg227_1
    del arg228_1
    buf397 = reinterpret_tensor(buf396, (1, 128, 2560), (327680, 2560, 1), 0); del buf396  # reuse
    buf398 = buf380; del buf380  # reuse
    buf399 = buf379; del buf379  # reuse
    buf401 = reinterpret_tensor(buf395, (1, 128, 2560), (327680, 2560, 1), 0); del buf395  # reuse
    cpp_fused_add_native_layer_norm_86(c_void_p(buf397.data_ptr()), c_void_p(buf333.data_ptr()), c_void_p(buf351.data_ptr()), c_void_p(buf371.data_ptr()), c_void_p(buf378.data_ptr()), c_void_p(arg229_1.data_ptr()), c_void_p(arg230_1.data_ptr()), c_void_p(buf398.data_ptr()), c_void_p(buf399.data_ptr()), c_void_p(buf401.data_ptr()))
    del arg229_1
    del arg230_1
    buf402 = buf378; del buf378  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_7_encoder_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg232_1, reinterpret_tensor(buf401, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg231_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf402)
    del arg231_1
    del arg232_1
    buf403 = reinterpret_tensor(buf401, (128, 2560), (2560, 1), 0); del buf401  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_7_encoder_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg234_1, reinterpret_tensor(buf81, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg233_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf403)
    del arg233_1
    del arg234_1
    buf404 = buf371; del buf371  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_7_encoder_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg236_1, reinterpret_tensor(buf81, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg235_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf404)
    del arg235_1
    del arg236_1
    buf405 = reinterpret_tensor(buf351, (1, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf351  # reuse
    buf406 = reinterpret_tensor(buf333, (1, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf333  # reuse
    buf407 = buf385; del buf385  # reuse
    cpp_fused_clone_87(c_void_p(buf402.data_ptr()), c_void_p(buf403.data_ptr()), c_void_p(buf404.data_ptr()), c_void_p(buf405.data_ptr()), c_void_p(buf406.data_ptr()), c_void_p(buf407.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf408 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf405, (1, 32, 128, 80), (0, 10240, 80, 1), 0), reinterpret_tensor(buf406, (1, 32, 128, 80), (0, 10240, 80, 1), 0), reinterpret_tensor(buf407, (1, 32, 128, 80), (0, 10240, 80, 1), 0), scale=1.0)
    buf409 = buf408[0]
    del buf408
    buf416 = reinterpret_tensor(buf409, (1, 128, 32, 80), (327680, 2560, 80, 1), 0); del buf409  # reuse
    cpp_fused_clone_88(c_void_p(buf416.data_ptr()))
    buf417 = reinterpret_tensor(buf407, (128, 2560), (2560, 1), 0); del buf407  # reuse
    # Source Nodes: [hidden_states_137], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg238_1, reinterpret_tensor(buf416, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg237_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf417)
    del arg237_1
    del arg238_1
    buf418 = buf399; del buf399  # reuse
    buf419 = buf398; del buf398  # reuse
    buf421 = reinterpret_tensor(buf416, (1, 128, 2560), (327680, 2560, 1), 0); del buf416  # reuse
    cpp_fused_add_native_layer_norm_89(c_void_p(buf397.data_ptr()), c_void_p(buf417.data_ptr()), c_void_p(arg239_1.data_ptr()), c_void_p(arg240_1.data_ptr()), c_void_p(buf418.data_ptr()), c_void_p(buf419.data_ptr()), c_void_p(buf421.data_ptr()))
    del arg239_1
    del arg240_1
    buf422 = reinterpret_tensor(buf377, (128, 10240), (10240, 1), 0); del buf377  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_7_fc1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg242_1, reinterpret_tensor(buf421, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg241_1, (2560, 10240), (1, 2560), 0), alpha=1, beta=1, out=buf422)
    del arg241_1
    del arg242_1
    buf423 = reinterpret_tensor(buf422, (1, 128, 10240), (1310720, 10240, 1), 0); del buf422  # reuse
    cpp_fused_gelu_90(c_void_p(buf423.data_ptr()))
    buf424 = reinterpret_tensor(buf421, (128, 2560), (2560, 1), 0); del buf421  # reuse
    # Source Nodes: [hidden_states_143], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg244_1, reinterpret_tensor(buf423, (128, 10240), (10240, 1), 0), reinterpret_tensor(arg243_1, (10240, 2560), (1, 10240), 0), alpha=1, beta=1, out=buf424)
    del arg243_1
    del arg244_1
    buf425 = buf419; del buf419  # reuse
    buf426 = buf418; del buf418  # reuse
    buf428 = reinterpret_tensor(buf406, (1, 128, 2560), (327680, 2560, 1), 0); del buf406  # reuse
    cpp_fused_add_native_layer_norm_91(c_void_p(buf397.data_ptr()), c_void_p(buf417.data_ptr()), c_void_p(buf424.data_ptr()), c_void_p(arg245_1.data_ptr()), c_void_p(arg246_1.data_ptr()), c_void_p(buf425.data_ptr()), c_void_p(buf426.data_ptr()), c_void_p(buf428.data_ptr()))
    del arg245_1
    del arg246_1
    buf429 = reinterpret_tensor(buf405, (128, 2560), (2560, 1), 0); del buf405  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_8_self_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg248_1, reinterpret_tensor(buf428, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg247_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf429)
    del arg247_1
    del arg248_1
    buf430 = buf404; del buf404  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_8_self_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg250_1, reinterpret_tensor(buf428, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg249_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf430)
    del arg249_1
    del arg250_1
    buf431 = reinterpret_tensor(buf403, (1, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf403  # reuse
    buf432 = reinterpret_tensor(buf402, (1, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf402  # reuse
    cpp_fused_clone_92(c_void_p(buf429.data_ptr()), c_void_p(buf430.data_ptr()), c_void_p(buf431.data_ptr()), c_void_p(buf432.data_ptr()))
    buf433 = buf392; del buf392  # reuse
    # Source Nodes: [attn_weights_52], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf431, (32, 128, 80), (10240, 80, 1), 0), reinterpret_tensor(buf432, (32, 80, 128), (10240, 1, 80), 0), out=buf433)
    buf434 = buf390; del buf390  # reuse
    buf435 = buf433; del buf433  # reuse
    buf436 = buf388; del buf388  # reuse
    cpp_fused__softmax_93(c_void_p(buf435.data_ptr()), c_void_p(buf434.data_ptr()), c_void_p(buf436.data_ptr()))
    buf437 = reinterpret_tensor(buf432, (128, 2560), (2560, 1), 0); del buf432  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_8_self_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg252_1, reinterpret_tensor(buf428, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg251_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf437)
    del arg251_1
    del arg252_1
    buf438 = buf435; del buf435  # reuse
    buf439 = reinterpret_tensor(buf428, (1, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf428  # reuse
    cpp_fused__softmax_clone_94(c_void_p(buf438.data_ptr()), c_void_p(buf436.data_ptr()), c_void_p(buf437.data_ptr()), c_void_p(buf439.data_ptr()))
    buf440 = reinterpret_tensor(buf437, (32, 128, 80), (10240, 80, 1), 0); del buf437  # reuse
    # Source Nodes: [attn_output_90, attn_weights_55], Original ATen: [aten._softmax, aten.bmm]
    extern_kernels.bmm(buf438, reinterpret_tensor(buf439, (32, 128, 80), (10240, 80, 1), 0), out=buf440)
    buf441 = reinterpret_tensor(buf439, (1, 128, 32, 80), (327680, 2560, 80, 1), 0); del buf439  # reuse
    cpp_fused_clone_95(c_void_p(buf440.data_ptr()), c_void_p(buf441.data_ptr()))
    buf442 = reinterpret_tensor(buf440, (128, 2560), (2560, 1), 0); del buf440  # reuse
    # Source Nodes: [hidden_states_148], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg254_1, reinterpret_tensor(buf441, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg253_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf442)
    del arg253_1
    del arg254_1
    buf443 = buf426; del buf426  # reuse
    buf444 = buf425; del buf425  # reuse
    buf446 = reinterpret_tensor(buf441, (1, 128, 2560), (327680, 2560, 1), 0); del buf441  # reuse
    cpp_fused_add_native_layer_norm_96(c_void_p(buf397.data_ptr()), c_void_p(buf417.data_ptr()), c_void_p(buf424.data_ptr()), c_void_p(buf442.data_ptr()), c_void_p(arg255_1.data_ptr()), c_void_p(arg256_1.data_ptr()), c_void_p(buf443.data_ptr()), c_void_p(buf444.data_ptr()), c_void_p(buf446.data_ptr()))
    del arg255_1
    del arg256_1
    buf447 = reinterpret_tensor(buf431, (128, 2560), (2560, 1), 0); del buf431  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_8_encoder_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg258_1, reinterpret_tensor(buf446, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg257_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf447)
    del arg257_1
    del arg258_1
    buf448 = reinterpret_tensor(buf446, (128, 2560), (2560, 1), 0); del buf446  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_8_encoder_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg260_1, reinterpret_tensor(buf81, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg259_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf448)
    del arg259_1
    del arg260_1
    buf449 = buf430; del buf430  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_8_encoder_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg262_1, reinterpret_tensor(buf81, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg261_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf449)
    del arg261_1
    del arg262_1
    buf450 = reinterpret_tensor(buf429, (1, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf429  # reuse
    buf451 = reinterpret_tensor(buf384, (1, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf384  # reuse
    buf452 = reinterpret_tensor(buf383, (1, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf383  # reuse
    cpp_fused_clone_97(c_void_p(buf447.data_ptr()), c_void_p(buf448.data_ptr()), c_void_p(buf449.data_ptr()), c_void_p(buf450.data_ptr()), c_void_p(buf451.data_ptr()), c_void_p(buf452.data_ptr()))
    del buf447
    del buf448
    # Source Nodes: [], Original ATen: []
    buf453 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf450, (1, 32, 128, 80), (0, 10240, 80, 1), 0), reinterpret_tensor(buf451, (1, 32, 128, 80), (0, 10240, 80, 1), 0), reinterpret_tensor(buf452, (1, 32, 128, 80), (0, 10240, 80, 1), 0), scale=1.0)
    buf454 = buf453[0]
    del buf453
    buf461 = reinterpret_tensor(buf454, (1, 128, 32, 80), (327680, 2560, 80, 1), 0); del buf454  # reuse
    cpp_fused_clone_98(c_void_p(buf461.data_ptr()))
    buf462 = reinterpret_tensor(buf452, (128, 2560), (2560, 1), 0); del buf452  # reuse
    # Source Nodes: [hidden_states_152], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg264_1, reinterpret_tensor(buf461, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg263_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf462)
    del arg263_1
    del arg264_1
    buf463 = reinterpret_tensor(buf462, (1, 128, 2560), (327680, 2560, 1), 0); del buf462  # reuse
    buf464 = buf444; del buf444  # reuse
    buf465 = buf443; del buf443  # reuse
    buf467 = reinterpret_tensor(buf461, (1, 128, 2560), (327680, 2560, 1), 0); del buf461  # reuse
    cpp_fused_add_native_layer_norm_99(c_void_p(buf463.data_ptr()), c_void_p(buf397.data_ptr()), c_void_p(buf417.data_ptr()), c_void_p(buf424.data_ptr()), c_void_p(buf442.data_ptr()), c_void_p(arg265_1.data_ptr()), c_void_p(arg266_1.data_ptr()), c_void_p(buf464.data_ptr()), c_void_p(buf465.data_ptr()), c_void_p(buf467.data_ptr()))
    del arg265_1
    del arg266_1
    buf468 = reinterpret_tensor(buf423, (128, 10240), (10240, 1), 0); del buf423  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_8_fc1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg268_1, reinterpret_tensor(buf467, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg267_1, (2560, 10240), (1, 2560), 0), alpha=1, beta=1, out=buf468)
    del arg267_1
    del arg268_1
    buf469 = reinterpret_tensor(buf468, (1, 128, 10240), (1310720, 10240, 1), 0); del buf468  # reuse
    cpp_fused_gelu_100(c_void_p(buf469.data_ptr()))
    buf470 = reinterpret_tensor(buf467, (128, 2560), (2560, 1), 0); del buf467  # reuse
    # Source Nodes: [hidden_states_158], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg270_1, reinterpret_tensor(buf469, (128, 10240), (10240, 1), 0), reinterpret_tensor(arg269_1, (10240, 2560), (1, 10240), 0), alpha=1, beta=1, out=buf470)
    del arg269_1
    del arg270_1
    buf471 = buf465; del buf465  # reuse
    buf472 = buf464; del buf464  # reuse
    buf474 = reinterpret_tensor(buf442, (1, 128, 2560), (327680, 2560, 1), 0); del buf442  # reuse
    cpp_fused_add_native_layer_norm_101(c_void_p(buf463.data_ptr()), c_void_p(buf470.data_ptr()), c_void_p(arg271_1.data_ptr()), c_void_p(arg272_1.data_ptr()), c_void_p(buf471.data_ptr()), c_void_p(buf472.data_ptr()), c_void_p(buf474.data_ptr()))
    del arg271_1
    del arg272_1
    buf475 = buf424; del buf424  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_9_self_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg274_1, reinterpret_tensor(buf474, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg273_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf475)
    del arg273_1
    del arg274_1
    buf476 = buf417; del buf417  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_9_self_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg276_1, reinterpret_tensor(buf474, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg275_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf476)
    del arg275_1
    del arg276_1
    buf477 = reinterpret_tensor(buf397, (1, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf397  # reuse
    buf478 = buf451; del buf451  # reuse
    cpp_fused_clone_102(c_void_p(buf475.data_ptr()), c_void_p(buf476.data_ptr()), c_void_p(buf477.data_ptr()), c_void_p(buf478.data_ptr()))
    buf479 = buf438; del buf438  # reuse
    # Source Nodes: [attn_weights_58], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf477, (32, 128, 80), (10240, 80, 1), 0), reinterpret_tensor(buf478, (32, 80, 128), (10240, 1, 80), 0), out=buf479)
    buf480 = buf436; del buf436  # reuse
    buf481 = buf479; del buf479  # reuse
    buf482 = buf434; del buf434  # reuse
    cpp_fused__softmax_103(c_void_p(buf481.data_ptr()), c_void_p(buf480.data_ptr()), c_void_p(buf482.data_ptr()))
    buf483 = reinterpret_tensor(buf478, (128, 2560), (2560, 1), 0); del buf478  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_9_self_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg278_1, reinterpret_tensor(buf474, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg277_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf483)
    del arg277_1
    del arg278_1
    buf484 = buf481; del buf481  # reuse
    buf485 = reinterpret_tensor(buf474, (1, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf474  # reuse
    cpp_fused__softmax_clone_104(c_void_p(buf484.data_ptr()), c_void_p(buf482.data_ptr()), c_void_p(buf483.data_ptr()), c_void_p(buf485.data_ptr()))
    buf486 = reinterpret_tensor(buf483, (32, 128, 80), (10240, 80, 1), 0); del buf483  # reuse
    # Source Nodes: [attn_output_100, attn_weights_61], Original ATen: [aten._softmax, aten.bmm]
    extern_kernels.bmm(buf484, reinterpret_tensor(buf485, (32, 128, 80), (10240, 80, 1), 0), out=buf486)
    buf487 = reinterpret_tensor(buf485, (1, 128, 32, 80), (327680, 2560, 80, 1), 0); del buf485  # reuse
    cpp_fused_clone_105(c_void_p(buf486.data_ptr()), c_void_p(buf487.data_ptr()))
    buf488 = reinterpret_tensor(buf486, (128, 2560), (2560, 1), 0); del buf486  # reuse
    # Source Nodes: [hidden_states_163], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg280_1, reinterpret_tensor(buf487, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg279_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf488)
    del arg279_1
    del arg280_1
    buf489 = buf472; del buf472  # reuse
    buf490 = buf471; del buf471  # reuse
    buf492 = reinterpret_tensor(buf487, (1, 128, 2560), (327680, 2560, 1), 0); del buf487  # reuse
    cpp_fused_add_native_layer_norm_106(c_void_p(buf463.data_ptr()), c_void_p(buf470.data_ptr()), c_void_p(buf488.data_ptr()), c_void_p(arg281_1.data_ptr()), c_void_p(arg282_1.data_ptr()), c_void_p(buf489.data_ptr()), c_void_p(buf490.data_ptr()), c_void_p(buf492.data_ptr()))
    del arg281_1
    del arg282_1
    buf493 = reinterpret_tensor(buf477, (128, 2560), (2560, 1), 0); del buf477  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_9_encoder_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg284_1, reinterpret_tensor(buf492, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg283_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf493)
    del arg283_1
    del arg284_1
    buf494 = reinterpret_tensor(buf492, (128, 2560), (2560, 1), 0); del buf492  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_9_encoder_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg286_1, reinterpret_tensor(buf81, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg285_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf494)
    del arg285_1
    del arg286_1
    buf495 = buf476; del buf476  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_9_encoder_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg288_1, reinterpret_tensor(buf81, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg287_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf495)
    del arg287_1
    del arg288_1
    buf496 = reinterpret_tensor(buf475, (1, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf475  # reuse
    buf497 = buf450; del buf450  # reuse
    buf498 = reinterpret_tensor(buf449, (1, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf449  # reuse
    cpp_fused_clone_107(c_void_p(buf493.data_ptr()), c_void_p(buf494.data_ptr()), c_void_p(buf495.data_ptr()), c_void_p(buf496.data_ptr()), c_void_p(buf497.data_ptr()), c_void_p(buf498.data_ptr()))
    del buf493
    del buf494
    # Source Nodes: [], Original ATen: []
    buf499 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf496, (1, 32, 128, 80), (0, 10240, 80, 1), 0), reinterpret_tensor(buf497, (1, 32, 128, 80), (0, 10240, 80, 1), 0), reinterpret_tensor(buf498, (1, 32, 128, 80), (0, 10240, 80, 1), 0), scale=1.0)
    buf500 = buf499[0]
    del buf499
    buf507 = reinterpret_tensor(buf500, (1, 128, 32, 80), (327680, 2560, 80, 1), 0); del buf500  # reuse
    cpp_fused_clone_108(c_void_p(buf507.data_ptr()))
    buf508 = reinterpret_tensor(buf498, (128, 2560), (2560, 1), 0); del buf498  # reuse
    # Source Nodes: [hidden_states_167], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg290_1, reinterpret_tensor(buf507, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg289_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf508)
    del arg289_1
    del arg290_1
    buf509 = buf490; del buf490  # reuse
    buf510 = buf489; del buf489  # reuse
    buf512 = reinterpret_tensor(buf507, (1, 128, 2560), (327680, 2560, 1), 0); del buf507  # reuse
    cpp_fused_add_native_layer_norm_109(c_void_p(buf463.data_ptr()), c_void_p(buf470.data_ptr()), c_void_p(buf488.data_ptr()), c_void_p(buf508.data_ptr()), c_void_p(arg291_1.data_ptr()), c_void_p(arg292_1.data_ptr()), c_void_p(buf509.data_ptr()), c_void_p(buf510.data_ptr()), c_void_p(buf512.data_ptr()))
    del arg291_1
    del arg292_1
    buf513 = reinterpret_tensor(buf469, (128, 10240), (10240, 1), 0); del buf469  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_9_fc1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg294_1, reinterpret_tensor(buf512, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg293_1, (2560, 10240), (1, 2560), 0), alpha=1, beta=1, out=buf513)
    del arg293_1
    del arg294_1
    buf514 = reinterpret_tensor(buf513, (1, 128, 10240), (1310720, 10240, 1), 0); del buf513  # reuse
    cpp_fused_gelu_110(c_void_p(buf514.data_ptr()))
    buf515 = reinterpret_tensor(buf512, (128, 2560), (2560, 1), 0); del buf512  # reuse
    # Source Nodes: [hidden_states_173], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg296_1, reinterpret_tensor(buf514, (128, 10240), (10240, 1), 0), reinterpret_tensor(arg295_1, (10240, 2560), (1, 10240), 0), alpha=1, beta=1, out=buf515)
    del arg295_1
    del arg296_1
    buf516 = reinterpret_tensor(buf515, (1, 128, 2560), (327680, 2560, 1), 0); del buf515  # reuse
    buf517 = buf510; del buf510  # reuse
    buf518 = buf509; del buf509  # reuse
    buf520 = reinterpret_tensor(buf497, (1, 128, 2560), (327680, 2560, 1), 0); del buf497  # reuse
    cpp_fused_add_native_layer_norm_111(c_void_p(buf516.data_ptr()), c_void_p(buf463.data_ptr()), c_void_p(buf470.data_ptr()), c_void_p(buf488.data_ptr()), c_void_p(buf508.data_ptr()), c_void_p(arg297_1.data_ptr()), c_void_p(arg298_1.data_ptr()), c_void_p(buf517.data_ptr()), c_void_p(buf518.data_ptr()), c_void_p(buf520.data_ptr()))
    del arg297_1
    del arg298_1
    buf521 = buf508; del buf508  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_10_self_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg300_1, reinterpret_tensor(buf520, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg299_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf521)
    del arg299_1
    del arg300_1
    buf522 = buf488; del buf488  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_10_self_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg302_1, reinterpret_tensor(buf520, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg301_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf522)
    del arg301_1
    del arg302_1
    buf523 = reinterpret_tensor(buf470, (1, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf470  # reuse
    buf524 = reinterpret_tensor(buf463, (1, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf463  # reuse
    cpp_fused_clone_112(c_void_p(buf521.data_ptr()), c_void_p(buf522.data_ptr()), c_void_p(buf523.data_ptr()), c_void_p(buf524.data_ptr()))
    buf525 = buf484; del buf484  # reuse
    # Source Nodes: [attn_weights_64], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf523, (32, 128, 80), (10240, 80, 1), 0), reinterpret_tensor(buf524, (32, 80, 128), (10240, 1, 80), 0), out=buf525)
    buf526 = buf482; del buf482  # reuse
    buf527 = buf525; del buf525  # reuse
    buf528 = buf480; del buf480  # reuse
    cpp_fused__softmax_113(c_void_p(buf527.data_ptr()), c_void_p(buf526.data_ptr()), c_void_p(buf528.data_ptr()))
    buf529 = reinterpret_tensor(buf524, (128, 2560), (2560, 1), 0); del buf524  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_10_self_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg304_1, reinterpret_tensor(buf520, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg303_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf529)
    del arg303_1
    del arg304_1
    buf530 = buf527; del buf527  # reuse
    buf531 = reinterpret_tensor(buf520, (1, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf520  # reuse
    cpp_fused__softmax_clone_114(c_void_p(buf530.data_ptr()), c_void_p(buf528.data_ptr()), c_void_p(buf529.data_ptr()), c_void_p(buf531.data_ptr()))
    buf532 = reinterpret_tensor(buf529, (32, 128, 80), (10240, 80, 1), 0); del buf529  # reuse
    # Source Nodes: [attn_output_110, attn_weights_67], Original ATen: [aten._softmax, aten.bmm]
    extern_kernels.bmm(buf530, reinterpret_tensor(buf531, (32, 128, 80), (10240, 80, 1), 0), out=buf532)
    buf533 = reinterpret_tensor(buf531, (1, 128, 32, 80), (327680, 2560, 80, 1), 0); del buf531  # reuse
    cpp_fused_clone_115(c_void_p(buf532.data_ptr()), c_void_p(buf533.data_ptr()))
    buf534 = reinterpret_tensor(buf532, (128, 2560), (2560, 1), 0); del buf532  # reuse
    # Source Nodes: [hidden_states_178], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg306_1, reinterpret_tensor(buf533, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg305_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf534)
    del arg305_1
    del arg306_1
    buf535 = buf518; del buf518  # reuse
    buf536 = buf517; del buf517  # reuse
    buf538 = reinterpret_tensor(buf533, (1, 128, 2560), (327680, 2560, 1), 0); del buf533  # reuse
    cpp_fused_add_native_layer_norm_116(c_void_p(buf516.data_ptr()), c_void_p(buf534.data_ptr()), c_void_p(arg307_1.data_ptr()), c_void_p(arg308_1.data_ptr()), c_void_p(buf535.data_ptr()), c_void_p(buf536.data_ptr()), c_void_p(buf538.data_ptr()))
    del arg307_1
    del arg308_1
    buf539 = reinterpret_tensor(buf523, (128, 2560), (2560, 1), 0); del buf523  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_10_encoder_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg310_1, reinterpret_tensor(buf538, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg309_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf539)
    del arg309_1
    del arg310_1
    buf540 = reinterpret_tensor(buf538, (128, 2560), (2560, 1), 0); del buf538  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_10_encoder_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg312_1, reinterpret_tensor(buf81, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg311_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf540)
    del arg311_1
    del arg312_1
    buf541 = buf522; del buf522  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_10_encoder_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg314_1, reinterpret_tensor(buf81, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg313_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf541)
    del arg313_1
    del arg314_1
    buf542 = reinterpret_tensor(buf521, (1, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf521  # reuse
    buf543 = buf496; del buf496  # reuse
    buf544 = reinterpret_tensor(buf495, (1, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf495  # reuse
    cpp_fused_clone_117(c_void_p(buf539.data_ptr()), c_void_p(buf540.data_ptr()), c_void_p(buf541.data_ptr()), c_void_p(buf542.data_ptr()), c_void_p(buf543.data_ptr()), c_void_p(buf544.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf545 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf542, (1, 32, 128, 80), (0, 10240, 80, 1), 0), reinterpret_tensor(buf543, (1, 32, 128, 80), (0, 10240, 80, 1), 0), reinterpret_tensor(buf544, (1, 32, 128, 80), (0, 10240, 80, 1), 0), scale=1.0)
    buf546 = buf545[0]
    del buf545
    buf553 = reinterpret_tensor(buf546, (1, 128, 32, 80), (327680, 2560, 80, 1), 0); del buf546  # reuse
    cpp_fused_clone_118(c_void_p(buf553.data_ptr()))
    buf554 = reinterpret_tensor(buf544, (128, 2560), (2560, 1), 0); del buf544  # reuse
    # Source Nodes: [hidden_states_182], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg316_1, reinterpret_tensor(buf553, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg315_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf554)
    del arg315_1
    del arg316_1
    buf555 = buf536; del buf536  # reuse
    buf556 = buf535; del buf535  # reuse
    buf558 = reinterpret_tensor(buf553, (1, 128, 2560), (327680, 2560, 1), 0); del buf553  # reuse
    cpp_fused_add_native_layer_norm_119(c_void_p(buf516.data_ptr()), c_void_p(buf534.data_ptr()), c_void_p(buf554.data_ptr()), c_void_p(arg317_1.data_ptr()), c_void_p(arg318_1.data_ptr()), c_void_p(buf555.data_ptr()), c_void_p(buf556.data_ptr()), c_void_p(buf558.data_ptr()))
    del arg317_1
    del arg318_1
    buf559 = reinterpret_tensor(buf514, (128, 10240), (10240, 1), 0); del buf514  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_10_fc1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg320_1, reinterpret_tensor(buf558, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg319_1, (2560, 10240), (1, 2560), 0), alpha=1, beta=1, out=buf559)
    del arg319_1
    del arg320_1
    buf560 = reinterpret_tensor(buf559, (1, 128, 10240), (1310720, 10240, 1), 0); del buf559  # reuse
    cpp_fused_gelu_120(c_void_p(buf560.data_ptr()))
    buf561 = reinterpret_tensor(buf558, (128, 2560), (2560, 1), 0); del buf558  # reuse
    # Source Nodes: [hidden_states_188], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg322_1, reinterpret_tensor(buf560, (128, 10240), (10240, 1), 0), reinterpret_tensor(arg321_1, (10240, 2560), (1, 10240), 0), alpha=1, beta=1, out=buf561)
    del arg321_1
    del arg322_1
    buf562 = buf556; del buf556  # reuse
    buf563 = buf555; del buf555  # reuse
    buf565 = reinterpret_tensor(buf543, (1, 128, 2560), (327680, 2560, 1), 0); del buf543  # reuse
    cpp_fused_add_native_layer_norm_121(c_void_p(buf516.data_ptr()), c_void_p(buf534.data_ptr()), c_void_p(buf554.data_ptr()), c_void_p(buf561.data_ptr()), c_void_p(arg323_1.data_ptr()), c_void_p(arg324_1.data_ptr()), c_void_p(buf562.data_ptr()), c_void_p(buf563.data_ptr()), c_void_p(buf565.data_ptr()))
    del arg323_1
    del arg324_1
    buf566 = reinterpret_tensor(buf542, (128, 2560), (2560, 1), 0); del buf542  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_11_self_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg326_1, reinterpret_tensor(buf565, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg325_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf566)
    del arg325_1
    del arg326_1
    buf567 = buf541; del buf541  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_11_self_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg328_1, reinterpret_tensor(buf565, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg327_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf567)
    del arg327_1
    del arg328_1
    buf568 = reinterpret_tensor(buf540, (1, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf540  # reuse
    buf569 = reinterpret_tensor(buf539, (1, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf539  # reuse
    cpp_fused_clone_122(c_void_p(buf566.data_ptr()), c_void_p(buf567.data_ptr()), c_void_p(buf568.data_ptr()), c_void_p(buf569.data_ptr()))
    buf570 = buf530; del buf530  # reuse
    # Source Nodes: [attn_weights_70], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf568, (32, 128, 80), (10240, 80, 1), 0), reinterpret_tensor(buf569, (32, 80, 128), (10240, 1, 80), 0), out=buf570)
    buf571 = buf528; del buf528  # reuse
    buf572 = buf570; del buf570  # reuse
    buf573 = buf526; del buf526  # reuse
    cpp_fused__softmax_123(c_void_p(buf572.data_ptr()), c_void_p(buf571.data_ptr()), c_void_p(buf573.data_ptr()))
    buf574 = reinterpret_tensor(buf569, (128, 2560), (2560, 1), 0); del buf569  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_11_self_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg330_1, reinterpret_tensor(buf565, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg329_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf574)
    del arg329_1
    del arg330_1
    buf575 = buf572; del buf572  # reuse
    buf576 = reinterpret_tensor(buf565, (1, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf565  # reuse
    cpp_fused__softmax_clone_124(c_void_p(buf575.data_ptr()), c_void_p(buf573.data_ptr()), c_void_p(buf574.data_ptr()), c_void_p(buf576.data_ptr()))
    buf577 = reinterpret_tensor(buf574, (32, 128, 80), (10240, 80, 1), 0); del buf574  # reuse
    # Source Nodes: [attn_output_120, attn_weights_73], Original ATen: [aten._softmax, aten.bmm]
    extern_kernels.bmm(buf575, reinterpret_tensor(buf576, (32, 128, 80), (10240, 80, 1), 0), out=buf577)
    buf578 = reinterpret_tensor(buf576, (1, 128, 32, 80), (327680, 2560, 80, 1), 0); del buf576  # reuse
    cpp_fused_clone_125(c_void_p(buf577.data_ptr()), c_void_p(buf578.data_ptr()))
    buf579 = reinterpret_tensor(buf577, (128, 2560), (2560, 1), 0); del buf577  # reuse
    # Source Nodes: [hidden_states_193], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg332_1, reinterpret_tensor(buf578, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg331_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf579)
    del arg331_1
    del arg332_1
    buf580 = reinterpret_tensor(buf579, (1, 128, 2560), (327680, 2560, 1), 0); del buf579  # reuse
    buf581 = buf563; del buf563  # reuse
    buf582 = buf562; del buf562  # reuse
    buf584 = reinterpret_tensor(buf578, (1, 128, 2560), (327680, 2560, 1), 0); del buf578  # reuse
    cpp_fused_add_native_layer_norm_126(c_void_p(buf580.data_ptr()), c_void_p(buf516.data_ptr()), c_void_p(buf534.data_ptr()), c_void_p(buf554.data_ptr()), c_void_p(buf561.data_ptr()), c_void_p(arg333_1.data_ptr()), c_void_p(arg334_1.data_ptr()), c_void_p(buf581.data_ptr()), c_void_p(buf582.data_ptr()), c_void_p(buf584.data_ptr()))
    del arg333_1
    del arg334_1
    buf585 = buf561; del buf561  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_11_encoder_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg336_1, reinterpret_tensor(buf584, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg335_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf585)
    del arg335_1
    del arg336_1
    buf586 = reinterpret_tensor(buf584, (128, 2560), (2560, 1), 0); del buf584  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_11_encoder_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg338_1, reinterpret_tensor(buf81, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg337_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf586)
    del arg337_1
    del arg338_1
    buf587 = buf554; del buf554  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_11_encoder_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg340_1, reinterpret_tensor(buf81, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg339_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf587)
    del arg339_1
    del arg340_1
    buf588 = reinterpret_tensor(buf534, (1, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf534  # reuse
    buf589 = reinterpret_tensor(buf516, (1, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf516  # reuse
    buf590 = buf568; del buf568  # reuse
    cpp_fused_clone_127(c_void_p(buf585.data_ptr()), c_void_p(buf586.data_ptr()), c_void_p(buf587.data_ptr()), c_void_p(buf588.data_ptr()), c_void_p(buf589.data_ptr()), c_void_p(buf590.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf591 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf588, (1, 32, 128, 80), (0, 10240, 80, 1), 0), reinterpret_tensor(buf589, (1, 32, 128, 80), (0, 10240, 80, 1), 0), reinterpret_tensor(buf590, (1, 32, 128, 80), (0, 10240, 80, 1), 0), scale=1.0)
    buf592 = buf591[0]
    del buf591
    buf599 = reinterpret_tensor(buf592, (1, 128, 32, 80), (327680, 2560, 80, 1), 0); del buf592  # reuse
    cpp_fused_clone_128(c_void_p(buf599.data_ptr()))
    buf600 = reinterpret_tensor(buf590, (128, 2560), (2560, 1), 0); del buf590  # reuse
    # Source Nodes: [hidden_states_197], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg342_1, reinterpret_tensor(buf599, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg341_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf600)
    del arg341_1
    del arg342_1
    buf601 = buf582; del buf582  # reuse
    buf602 = buf581; del buf581  # reuse
    buf604 = reinterpret_tensor(buf599, (1, 128, 2560), (327680, 2560, 1), 0); del buf599  # reuse
    cpp_fused_add_native_layer_norm_129(c_void_p(buf580.data_ptr()), c_void_p(buf600.data_ptr()), c_void_p(arg343_1.data_ptr()), c_void_p(arg344_1.data_ptr()), c_void_p(buf601.data_ptr()), c_void_p(buf602.data_ptr()), c_void_p(buf604.data_ptr()))
    del arg343_1
    del arg344_1
    buf605 = reinterpret_tensor(buf560, (128, 10240), (10240, 1), 0); del buf560  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_11_fc1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg346_1, reinterpret_tensor(buf604, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg345_1, (2560, 10240), (1, 2560), 0), alpha=1, beta=1, out=buf605)
    del arg345_1
    del arg346_1
    buf606 = reinterpret_tensor(buf605, (1, 128, 10240), (1310720, 10240, 1), 0); del buf605  # reuse
    cpp_fused_gelu_130(c_void_p(buf606.data_ptr()))
    buf607 = reinterpret_tensor(buf604, (128, 2560), (2560, 1), 0); del buf604  # reuse
    # Source Nodes: [hidden_states_203], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg348_1, reinterpret_tensor(buf606, (128, 10240), (10240, 1), 0), reinterpret_tensor(arg347_1, (10240, 2560), (1, 10240), 0), alpha=1, beta=1, out=buf607)
    del arg347_1
    del arg348_1
    buf608 = buf602; del buf602  # reuse
    buf609 = buf601; del buf601  # reuse
    buf611 = reinterpret_tensor(buf589, (1, 128, 2560), (327680, 2560, 1), 0); del buf589  # reuse
    cpp_fused_add_native_layer_norm_131(c_void_p(buf580.data_ptr()), c_void_p(buf600.data_ptr()), c_void_p(buf607.data_ptr()), c_void_p(arg349_1.data_ptr()), c_void_p(arg350_1.data_ptr()), c_void_p(buf608.data_ptr()), c_void_p(buf609.data_ptr()), c_void_p(buf611.data_ptr()))
    del arg349_1
    del arg350_1
    buf612 = reinterpret_tensor(buf588, (128, 2560), (2560, 1), 0); del buf588  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_12_self_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg352_1, reinterpret_tensor(buf611, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg351_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf612)
    del arg351_1
    del arg352_1
    buf613 = buf587; del buf587  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_12_self_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg354_1, reinterpret_tensor(buf611, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg353_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf613)
    del arg353_1
    del arg354_1
    buf614 = reinterpret_tensor(buf586, (1, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf586  # reuse
    buf615 = reinterpret_tensor(buf585, (1, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf585  # reuse
    cpp_fused_clone_132(c_void_p(buf612.data_ptr()), c_void_p(buf613.data_ptr()), c_void_p(buf614.data_ptr()), c_void_p(buf615.data_ptr()))
    buf616 = buf575; del buf575  # reuse
    # Source Nodes: [attn_weights_76], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf614, (32, 128, 80), (10240, 80, 1), 0), reinterpret_tensor(buf615, (32, 80, 128), (10240, 1, 80), 0), out=buf616)
    buf617 = buf573; del buf573  # reuse
    buf618 = buf616; del buf616  # reuse
    buf619 = buf571; del buf571  # reuse
    cpp_fused__softmax_133(c_void_p(buf618.data_ptr()), c_void_p(buf617.data_ptr()), c_void_p(buf619.data_ptr()))
    buf620 = reinterpret_tensor(buf615, (128, 2560), (2560, 1), 0); del buf615  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_12_self_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg356_1, reinterpret_tensor(buf611, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg355_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf620)
    del arg355_1
    del arg356_1
    buf621 = buf618; del buf618  # reuse
    buf622 = reinterpret_tensor(buf611, (1, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf611  # reuse
    cpp_fused__softmax_clone_134(c_void_p(buf621.data_ptr()), c_void_p(buf619.data_ptr()), c_void_p(buf620.data_ptr()), c_void_p(buf622.data_ptr()))
    buf623 = reinterpret_tensor(buf620, (32, 128, 80), (10240, 80, 1), 0); del buf620  # reuse
    # Source Nodes: [attn_output_130, attn_weights_79], Original ATen: [aten._softmax, aten.bmm]
    extern_kernels.bmm(buf621, reinterpret_tensor(buf622, (32, 128, 80), (10240, 80, 1), 0), out=buf623)
    buf624 = reinterpret_tensor(buf622, (1, 128, 32, 80), (327680, 2560, 80, 1), 0); del buf622  # reuse
    cpp_fused_clone_135(c_void_p(buf623.data_ptr()), c_void_p(buf624.data_ptr()))
    buf625 = reinterpret_tensor(buf623, (128, 2560), (2560, 1), 0); del buf623  # reuse
    # Source Nodes: [hidden_states_208], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg358_1, reinterpret_tensor(buf624, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg357_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf625)
    del arg357_1
    del arg358_1
    buf626 = buf609; del buf609  # reuse
    buf627 = buf608; del buf608  # reuse
    buf629 = reinterpret_tensor(buf624, (1, 128, 2560), (327680, 2560, 1), 0); del buf624  # reuse
    cpp_fused_add_native_layer_norm_136(c_void_p(buf580.data_ptr()), c_void_p(buf600.data_ptr()), c_void_p(buf607.data_ptr()), c_void_p(buf625.data_ptr()), c_void_p(arg359_1.data_ptr()), c_void_p(arg360_1.data_ptr()), c_void_p(buf626.data_ptr()), c_void_p(buf627.data_ptr()), c_void_p(buf629.data_ptr()))
    del arg359_1
    del arg360_1
    buf630 = reinterpret_tensor(buf614, (128, 2560), (2560, 1), 0); del buf614  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_12_encoder_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg362_1, reinterpret_tensor(buf629, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg361_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf630)
    del arg361_1
    del arg362_1
    buf631 = reinterpret_tensor(buf629, (128, 2560), (2560, 1), 0); del buf629  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_12_encoder_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg364_1, reinterpret_tensor(buf81, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg363_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf631)
    del arg363_1
    del arg364_1
    buf632 = buf613; del buf613  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_12_encoder_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg366_1, reinterpret_tensor(buf81, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg365_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf632)
    del arg365_1
    del arg366_1
    buf633 = reinterpret_tensor(buf612, (1, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf612  # reuse
    buf634 = reinterpret_tensor(buf567, (1, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf567  # reuse
    buf635 = reinterpret_tensor(buf566, (1, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf566  # reuse
    cpp_fused_clone_137(c_void_p(buf630.data_ptr()), c_void_p(buf631.data_ptr()), c_void_p(buf632.data_ptr()), c_void_p(buf633.data_ptr()), c_void_p(buf634.data_ptr()), c_void_p(buf635.data_ptr()))
    del buf630
    del buf631
    # Source Nodes: [], Original ATen: []
    buf636 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf633, (1, 32, 128, 80), (0, 10240, 80, 1), 0), reinterpret_tensor(buf634, (1, 32, 128, 80), (0, 10240, 80, 1), 0), reinterpret_tensor(buf635, (1, 32, 128, 80), (0, 10240, 80, 1), 0), scale=1.0)
    buf637 = buf636[0]
    del buf636
    buf644 = reinterpret_tensor(buf637, (1, 128, 32, 80), (327680, 2560, 80, 1), 0); del buf637  # reuse
    cpp_fused_clone_138(c_void_p(buf644.data_ptr()))
    buf645 = reinterpret_tensor(buf635, (128, 2560), (2560, 1), 0); del buf635  # reuse
    # Source Nodes: [hidden_states_212], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg368_1, reinterpret_tensor(buf644, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg367_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf645)
    del arg367_1
    del arg368_1
    buf646 = reinterpret_tensor(buf645, (1, 128, 2560), (327680, 2560, 1), 0); del buf645  # reuse
    buf647 = buf627; del buf627  # reuse
    buf648 = buf626; del buf626  # reuse
    buf650 = reinterpret_tensor(buf644, (1, 128, 2560), (327680, 2560, 1), 0); del buf644  # reuse
    cpp_fused_add_native_layer_norm_139(c_void_p(buf646.data_ptr()), c_void_p(buf580.data_ptr()), c_void_p(buf600.data_ptr()), c_void_p(buf607.data_ptr()), c_void_p(buf625.data_ptr()), c_void_p(arg369_1.data_ptr()), c_void_p(arg370_1.data_ptr()), c_void_p(buf647.data_ptr()), c_void_p(buf648.data_ptr()), c_void_p(buf650.data_ptr()))
    del arg369_1
    del arg370_1
    buf651 = reinterpret_tensor(buf606, (128, 10240), (10240, 1), 0); del buf606  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_12_fc1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg372_1, reinterpret_tensor(buf650, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg371_1, (2560, 10240), (1, 2560), 0), alpha=1, beta=1, out=buf651)
    del arg371_1
    del arg372_1
    buf652 = reinterpret_tensor(buf651, (1, 128, 10240), (1310720, 10240, 1), 0); del buf651  # reuse
    cpp_fused_gelu_140(c_void_p(buf652.data_ptr()))
    buf653 = reinterpret_tensor(buf650, (128, 2560), (2560, 1), 0); del buf650  # reuse
    # Source Nodes: [hidden_states_218], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg374_1, reinterpret_tensor(buf652, (128, 10240), (10240, 1), 0), reinterpret_tensor(arg373_1, (10240, 2560), (1, 10240), 0), alpha=1, beta=1, out=buf653)
    del arg373_1
    del arg374_1
    buf654 = buf648; del buf648  # reuse
    buf655 = buf647; del buf647  # reuse
    buf657 = reinterpret_tensor(buf625, (1, 128, 2560), (327680, 2560, 1), 0); del buf625  # reuse
    cpp_fused_add_native_layer_norm_141(c_void_p(buf646.data_ptr()), c_void_p(buf653.data_ptr()), c_void_p(arg375_1.data_ptr()), c_void_p(arg376_1.data_ptr()), c_void_p(buf654.data_ptr()), c_void_p(buf655.data_ptr()), c_void_p(buf657.data_ptr()))
    del arg375_1
    del arg376_1
    buf658 = buf607; del buf607  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_13_self_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg378_1, reinterpret_tensor(buf657, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg377_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf658)
    del arg377_1
    del arg378_1
    buf659 = buf600; del buf600  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_13_self_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg380_1, reinterpret_tensor(buf657, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg379_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf659)
    del arg379_1
    del arg380_1
    buf660 = reinterpret_tensor(buf580, (1, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf580  # reuse
    buf661 = buf634; del buf634  # reuse
    cpp_fused_clone_142(c_void_p(buf658.data_ptr()), c_void_p(buf659.data_ptr()), c_void_p(buf660.data_ptr()), c_void_p(buf661.data_ptr()))
    buf662 = buf621; del buf621  # reuse
    # Source Nodes: [attn_weights_82], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf660, (32, 128, 80), (10240, 80, 1), 0), reinterpret_tensor(buf661, (32, 80, 128), (10240, 1, 80), 0), out=buf662)
    buf663 = buf619; del buf619  # reuse
    buf664 = buf662; del buf662  # reuse
    buf665 = buf617; del buf617  # reuse
    cpp_fused__softmax_143(c_void_p(buf664.data_ptr()), c_void_p(buf663.data_ptr()), c_void_p(buf665.data_ptr()))
    buf666 = reinterpret_tensor(buf661, (128, 2560), (2560, 1), 0); del buf661  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_13_self_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg382_1, reinterpret_tensor(buf657, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg381_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf666)
    del arg381_1
    del arg382_1
    buf667 = buf664; del buf664  # reuse
    buf668 = reinterpret_tensor(buf657, (1, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf657  # reuse
    cpp_fused__softmax_clone_144(c_void_p(buf667.data_ptr()), c_void_p(buf665.data_ptr()), c_void_p(buf666.data_ptr()), c_void_p(buf668.data_ptr()))
    buf669 = reinterpret_tensor(buf666, (32, 128, 80), (10240, 80, 1), 0); del buf666  # reuse
    # Source Nodes: [attn_output_140, attn_weights_85], Original ATen: [aten._softmax, aten.bmm]
    extern_kernels.bmm(buf667, reinterpret_tensor(buf668, (32, 128, 80), (10240, 80, 1), 0), out=buf669)
    buf670 = reinterpret_tensor(buf668, (1, 128, 32, 80), (327680, 2560, 80, 1), 0); del buf668  # reuse
    cpp_fused_clone_145(c_void_p(buf669.data_ptr()), c_void_p(buf670.data_ptr()))
    buf671 = reinterpret_tensor(buf669, (128, 2560), (2560, 1), 0); del buf669  # reuse
    # Source Nodes: [hidden_states_223], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg384_1, reinterpret_tensor(buf670, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg383_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf671)
    del arg383_1
    del arg384_1
    buf672 = buf655; del buf655  # reuse
    buf673 = buf654; del buf654  # reuse
    buf675 = reinterpret_tensor(buf670, (1, 128, 2560), (327680, 2560, 1), 0); del buf670  # reuse
    cpp_fused_add_native_layer_norm_146(c_void_p(buf646.data_ptr()), c_void_p(buf653.data_ptr()), c_void_p(buf671.data_ptr()), c_void_p(arg385_1.data_ptr()), c_void_p(arg386_1.data_ptr()), c_void_p(buf672.data_ptr()), c_void_p(buf673.data_ptr()), c_void_p(buf675.data_ptr()))
    del arg385_1
    del arg386_1
    buf676 = reinterpret_tensor(buf660, (128, 2560), (2560, 1), 0); del buf660  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_13_encoder_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg388_1, reinterpret_tensor(buf675, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg387_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf676)
    del arg387_1
    del arg388_1
    buf677 = reinterpret_tensor(buf675, (128, 2560), (2560, 1), 0); del buf675  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_13_encoder_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg390_1, reinterpret_tensor(buf81, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg389_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf677)
    del arg389_1
    del arg390_1
    buf678 = buf659; del buf659  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_13_encoder_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg392_1, reinterpret_tensor(buf81, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg391_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf678)
    del arg391_1
    del arg392_1
    buf679 = reinterpret_tensor(buf658, (1, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf658  # reuse
    buf680 = buf633; del buf633  # reuse
    buf681 = reinterpret_tensor(buf632, (1, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf632  # reuse
    cpp_fused_clone_147(c_void_p(buf676.data_ptr()), c_void_p(buf677.data_ptr()), c_void_p(buf678.data_ptr()), c_void_p(buf679.data_ptr()), c_void_p(buf680.data_ptr()), c_void_p(buf681.data_ptr()))
    del buf676
    del buf677
    # Source Nodes: [], Original ATen: []
    buf682 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf679, (1, 32, 128, 80), (0, 10240, 80, 1), 0), reinterpret_tensor(buf680, (1, 32, 128, 80), (0, 10240, 80, 1), 0), reinterpret_tensor(buf681, (1, 32, 128, 80), (0, 10240, 80, 1), 0), scale=1.0)
    buf683 = buf682[0]
    del buf682
    buf690 = reinterpret_tensor(buf683, (1, 128, 32, 80), (327680, 2560, 80, 1), 0); del buf683  # reuse
    cpp_fused_clone_148(c_void_p(buf690.data_ptr()))
    buf691 = reinterpret_tensor(buf681, (128, 2560), (2560, 1), 0); del buf681  # reuse
    # Source Nodes: [hidden_states_227], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg394_1, reinterpret_tensor(buf690, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg393_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf691)
    del arg393_1
    del arg394_1
    buf692 = buf673; del buf673  # reuse
    buf693 = buf672; del buf672  # reuse
    buf695 = reinterpret_tensor(buf690, (1, 128, 2560), (327680, 2560, 1), 0); del buf690  # reuse
    cpp_fused_add_native_layer_norm_149(c_void_p(buf646.data_ptr()), c_void_p(buf653.data_ptr()), c_void_p(buf671.data_ptr()), c_void_p(buf691.data_ptr()), c_void_p(arg395_1.data_ptr()), c_void_p(arg396_1.data_ptr()), c_void_p(buf692.data_ptr()), c_void_p(buf693.data_ptr()), c_void_p(buf695.data_ptr()))
    del arg395_1
    del arg396_1
    buf696 = reinterpret_tensor(buf652, (128, 10240), (10240, 1), 0); del buf652  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_13_fc1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg398_1, reinterpret_tensor(buf695, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg397_1, (2560, 10240), (1, 2560), 0), alpha=1, beta=1, out=buf696)
    del arg397_1
    del arg398_1
    buf697 = reinterpret_tensor(buf696, (1, 128, 10240), (1310720, 10240, 1), 0); del buf696  # reuse
    cpp_fused_gelu_150(c_void_p(buf697.data_ptr()))
    buf698 = reinterpret_tensor(buf695, (128, 2560), (2560, 1), 0); del buf695  # reuse
    # Source Nodes: [hidden_states_233], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg400_1, reinterpret_tensor(buf697, (128, 10240), (10240, 1), 0), reinterpret_tensor(arg399_1, (10240, 2560), (1, 10240), 0), alpha=1, beta=1, out=buf698)
    del arg399_1
    del arg400_1
    buf699 = reinterpret_tensor(buf698, (1, 128, 2560), (327680, 2560, 1), 0); del buf698  # reuse
    buf700 = buf693; del buf693  # reuse
    buf701 = buf692; del buf692  # reuse
    buf703 = reinterpret_tensor(buf680, (1, 128, 2560), (327680, 2560, 1), 0); del buf680  # reuse
    cpp_fused_add_native_layer_norm_151(c_void_p(buf699.data_ptr()), c_void_p(buf646.data_ptr()), c_void_p(buf653.data_ptr()), c_void_p(buf671.data_ptr()), c_void_p(buf691.data_ptr()), c_void_p(arg401_1.data_ptr()), c_void_p(arg402_1.data_ptr()), c_void_p(buf700.data_ptr()), c_void_p(buf701.data_ptr()), c_void_p(buf703.data_ptr()))
    del arg401_1
    del arg402_1
    buf704 = buf691; del buf691  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_14_self_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg404_1, reinterpret_tensor(buf703, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg403_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf704)
    del arg403_1
    del arg404_1
    buf705 = buf671; del buf671  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_14_self_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg406_1, reinterpret_tensor(buf703, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg405_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf705)
    del arg405_1
    del arg406_1
    buf706 = reinterpret_tensor(buf653, (1, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf653  # reuse
    buf707 = reinterpret_tensor(buf646, (1, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf646  # reuse
    cpp_fused_clone_152(c_void_p(buf704.data_ptr()), c_void_p(buf705.data_ptr()), c_void_p(buf706.data_ptr()), c_void_p(buf707.data_ptr()))
    buf708 = buf667; del buf667  # reuse
    # Source Nodes: [attn_weights_88], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf706, (32, 128, 80), (10240, 80, 1), 0), reinterpret_tensor(buf707, (32, 80, 128), (10240, 1, 80), 0), out=buf708)
    buf709 = buf665; del buf665  # reuse
    buf710 = buf708; del buf708  # reuse
    buf711 = buf663; del buf663  # reuse
    cpp_fused__softmax_153(c_void_p(buf710.data_ptr()), c_void_p(buf709.data_ptr()), c_void_p(buf711.data_ptr()))
    buf712 = reinterpret_tensor(buf707, (128, 2560), (2560, 1), 0); del buf707  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_14_self_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg408_1, reinterpret_tensor(buf703, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg407_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf712)
    del arg407_1
    del arg408_1
    buf713 = buf710; del buf710  # reuse
    buf714 = reinterpret_tensor(buf703, (1, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf703  # reuse
    cpp_fused__softmax_clone_154(c_void_p(buf713.data_ptr()), c_void_p(buf711.data_ptr()), c_void_p(buf712.data_ptr()), c_void_p(buf714.data_ptr()))
    buf715 = reinterpret_tensor(buf712, (32, 128, 80), (10240, 80, 1), 0); del buf712  # reuse
    # Source Nodes: [attn_output_150, attn_weights_91], Original ATen: [aten._softmax, aten.bmm]
    extern_kernels.bmm(buf713, reinterpret_tensor(buf714, (32, 128, 80), (10240, 80, 1), 0), out=buf715)
    buf716 = reinterpret_tensor(buf714, (1, 128, 32, 80), (327680, 2560, 80, 1), 0); del buf714  # reuse
    cpp_fused_clone_155(c_void_p(buf715.data_ptr()), c_void_p(buf716.data_ptr()))
    buf717 = reinterpret_tensor(buf715, (128, 2560), (2560, 1), 0); del buf715  # reuse
    # Source Nodes: [hidden_states_238], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg410_1, reinterpret_tensor(buf716, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg409_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf717)
    del arg409_1
    del arg410_1
    buf718 = buf701; del buf701  # reuse
    buf719 = buf700; del buf700  # reuse
    buf721 = reinterpret_tensor(buf716, (1, 128, 2560), (327680, 2560, 1), 0); del buf716  # reuse
    cpp_fused_add_native_layer_norm_156(c_void_p(buf699.data_ptr()), c_void_p(buf717.data_ptr()), c_void_p(arg411_1.data_ptr()), c_void_p(arg412_1.data_ptr()), c_void_p(buf718.data_ptr()), c_void_p(buf719.data_ptr()), c_void_p(buf721.data_ptr()))
    del arg411_1
    del arg412_1
    buf722 = reinterpret_tensor(buf706, (128, 2560), (2560, 1), 0); del buf706  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_14_encoder_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg414_1, reinterpret_tensor(buf721, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg413_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf722)
    del arg413_1
    del arg414_1
    buf723 = reinterpret_tensor(buf721, (128, 2560), (2560, 1), 0); del buf721  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_14_encoder_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg416_1, reinterpret_tensor(buf81, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg415_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf723)
    del arg415_1
    del arg416_1
    buf724 = buf705; del buf705  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_14_encoder_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg418_1, reinterpret_tensor(buf81, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg417_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf724)
    del arg417_1
    del arg418_1
    buf725 = reinterpret_tensor(buf704, (1, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf704  # reuse
    buf726 = buf679; del buf679  # reuse
    buf727 = reinterpret_tensor(buf678, (1, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf678  # reuse
    cpp_fused_clone_157(c_void_p(buf722.data_ptr()), c_void_p(buf723.data_ptr()), c_void_p(buf724.data_ptr()), c_void_p(buf725.data_ptr()), c_void_p(buf726.data_ptr()), c_void_p(buf727.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf728 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf725, (1, 32, 128, 80), (0, 10240, 80, 1), 0), reinterpret_tensor(buf726, (1, 32, 128, 80), (0, 10240, 80, 1), 0), reinterpret_tensor(buf727, (1, 32, 128, 80), (0, 10240, 80, 1), 0), scale=1.0)
    buf729 = buf728[0]
    del buf728
    buf736 = reinterpret_tensor(buf729, (1, 128, 32, 80), (327680, 2560, 80, 1), 0); del buf729  # reuse
    cpp_fused_clone_158(c_void_p(buf736.data_ptr()))
    buf737 = reinterpret_tensor(buf727, (128, 2560), (2560, 1), 0); del buf727  # reuse
    # Source Nodes: [hidden_states_242], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg420_1, reinterpret_tensor(buf736, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg419_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf737)
    del arg419_1
    del arg420_1
    buf738 = buf719; del buf719  # reuse
    buf739 = buf718; del buf718  # reuse
    buf741 = reinterpret_tensor(buf736, (1, 128, 2560), (327680, 2560, 1), 0); del buf736  # reuse
    cpp_fused_add_native_layer_norm_159(c_void_p(buf699.data_ptr()), c_void_p(buf717.data_ptr()), c_void_p(buf737.data_ptr()), c_void_p(arg421_1.data_ptr()), c_void_p(arg422_1.data_ptr()), c_void_p(buf738.data_ptr()), c_void_p(buf739.data_ptr()), c_void_p(buf741.data_ptr()))
    del arg421_1
    del arg422_1
    buf742 = reinterpret_tensor(buf697, (128, 10240), (10240, 1), 0); del buf697  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_14_fc1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg424_1, reinterpret_tensor(buf741, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg423_1, (2560, 10240), (1, 2560), 0), alpha=1, beta=1, out=buf742)
    del arg423_1
    del arg424_1
    buf743 = reinterpret_tensor(buf742, (1, 128, 10240), (1310720, 10240, 1), 0); del buf742  # reuse
    cpp_fused_gelu_160(c_void_p(buf743.data_ptr()))
    buf744 = reinterpret_tensor(buf741, (128, 2560), (2560, 1), 0); del buf741  # reuse
    # Source Nodes: [hidden_states_248], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg426_1, reinterpret_tensor(buf743, (128, 10240), (10240, 1), 0), reinterpret_tensor(arg425_1, (10240, 2560), (1, 10240), 0), alpha=1, beta=1, out=buf744)
    del arg425_1
    del arg426_1
    buf745 = buf739; del buf739  # reuse
    buf746 = buf738; del buf738  # reuse
    buf748 = reinterpret_tensor(buf726, (1, 128, 2560), (327680, 2560, 1), 0); del buf726  # reuse
    cpp_fused_add_native_layer_norm_161(c_void_p(buf699.data_ptr()), c_void_p(buf717.data_ptr()), c_void_p(buf737.data_ptr()), c_void_p(buf744.data_ptr()), c_void_p(arg427_1.data_ptr()), c_void_p(arg428_1.data_ptr()), c_void_p(buf745.data_ptr()), c_void_p(buf746.data_ptr()), c_void_p(buf748.data_ptr()))
    del arg427_1
    del arg428_1
    buf749 = reinterpret_tensor(buf725, (128, 2560), (2560, 1), 0); del buf725  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_15_self_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg430_1, reinterpret_tensor(buf748, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg429_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf749)
    del arg429_1
    del arg430_1
    buf750 = buf724; del buf724  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_15_self_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg432_1, reinterpret_tensor(buf748, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg431_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf750)
    del arg431_1
    del arg432_1
    buf751 = reinterpret_tensor(buf723, (1, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf723  # reuse
    buf752 = reinterpret_tensor(buf722, (1, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf722  # reuse
    cpp_fused_clone_162(c_void_p(buf749.data_ptr()), c_void_p(buf750.data_ptr()), c_void_p(buf751.data_ptr()), c_void_p(buf752.data_ptr()))
    buf753 = buf713; del buf713  # reuse
    # Source Nodes: [attn_weights_94], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf751, (32, 128, 80), (10240, 80, 1), 0), reinterpret_tensor(buf752, (32, 80, 128), (10240, 1, 80), 0), out=buf753)
    buf754 = buf711; del buf711  # reuse
    buf755 = buf753; del buf753  # reuse
    buf756 = buf709; del buf709  # reuse
    cpp_fused__softmax_163(c_void_p(buf755.data_ptr()), c_void_p(buf754.data_ptr()), c_void_p(buf756.data_ptr()))
    buf757 = reinterpret_tensor(buf752, (128, 2560), (2560, 1), 0); del buf752  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_15_self_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg434_1, reinterpret_tensor(buf748, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg433_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf757)
    del arg433_1
    del arg434_1
    buf758 = buf755; del buf755  # reuse
    buf759 = reinterpret_tensor(buf748, (1, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf748  # reuse
    cpp_fused__softmax_clone_164(c_void_p(buf758.data_ptr()), c_void_p(buf756.data_ptr()), c_void_p(buf757.data_ptr()), c_void_p(buf759.data_ptr()))
    buf760 = reinterpret_tensor(buf757, (32, 128, 80), (10240, 80, 1), 0); del buf757  # reuse
    # Source Nodes: [attn_output_160, attn_weights_97], Original ATen: [aten._softmax, aten.bmm]
    extern_kernels.bmm(buf758, reinterpret_tensor(buf759, (32, 128, 80), (10240, 80, 1), 0), out=buf760)
    buf761 = reinterpret_tensor(buf759, (1, 128, 32, 80), (327680, 2560, 80, 1), 0); del buf759  # reuse
    cpp_fused_clone_165(c_void_p(buf760.data_ptr()), c_void_p(buf761.data_ptr()))
    buf762 = reinterpret_tensor(buf760, (128, 2560), (2560, 1), 0); del buf760  # reuse
    # Source Nodes: [hidden_states_253], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg436_1, reinterpret_tensor(buf761, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg435_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf762)
    del arg435_1
    del arg436_1
    buf763 = reinterpret_tensor(buf762, (1, 128, 2560), (327680, 2560, 1), 0); del buf762  # reuse
    buf764 = buf746; del buf746  # reuse
    buf765 = buf745; del buf745  # reuse
    buf767 = reinterpret_tensor(buf761, (1, 128, 2560), (327680, 2560, 1), 0); del buf761  # reuse
    cpp_fused_add_native_layer_norm_166(c_void_p(buf763.data_ptr()), c_void_p(buf699.data_ptr()), c_void_p(buf717.data_ptr()), c_void_p(buf737.data_ptr()), c_void_p(buf744.data_ptr()), c_void_p(arg437_1.data_ptr()), c_void_p(arg438_1.data_ptr()), c_void_p(buf764.data_ptr()), c_void_p(buf765.data_ptr()), c_void_p(buf767.data_ptr()))
    del arg437_1
    del arg438_1
    buf768 = buf744; del buf744  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_15_encoder_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg440_1, reinterpret_tensor(buf767, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg439_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf768)
    del arg439_1
    del arg440_1
    buf769 = reinterpret_tensor(buf767, (128, 2560), (2560, 1), 0); del buf767  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_15_encoder_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg442_1, reinterpret_tensor(buf81, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg441_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf769)
    del arg441_1
    del arg442_1
    buf770 = buf737; del buf737  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_15_encoder_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg444_1, reinterpret_tensor(buf81, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg443_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf770)
    del arg443_1
    del arg444_1
    buf771 = reinterpret_tensor(buf717, (1, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf717  # reuse
    buf772 = reinterpret_tensor(buf699, (1, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf699  # reuse
    buf773 = buf751; del buf751  # reuse
    cpp_fused_clone_167(c_void_p(buf768.data_ptr()), c_void_p(buf769.data_ptr()), c_void_p(buf770.data_ptr()), c_void_p(buf771.data_ptr()), c_void_p(buf772.data_ptr()), c_void_p(buf773.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf774 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf771, (1, 32, 128, 80), (0, 10240, 80, 1), 0), reinterpret_tensor(buf772, (1, 32, 128, 80), (0, 10240, 80, 1), 0), reinterpret_tensor(buf773, (1, 32, 128, 80), (0, 10240, 80, 1), 0), scale=1.0)
    buf775 = buf774[0]
    del buf774
    buf782 = reinterpret_tensor(buf775, (1, 128, 32, 80), (327680, 2560, 80, 1), 0); del buf775  # reuse
    cpp_fused_clone_168(c_void_p(buf782.data_ptr()))
    buf783 = reinterpret_tensor(buf773, (128, 2560), (2560, 1), 0); del buf773  # reuse
    # Source Nodes: [hidden_states_257], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg446_1, reinterpret_tensor(buf782, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg445_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf783)
    del arg445_1
    del arg446_1
    buf784 = buf765; del buf765  # reuse
    buf785 = buf764; del buf764  # reuse
    buf787 = reinterpret_tensor(buf782, (1, 128, 2560), (327680, 2560, 1), 0); del buf782  # reuse
    cpp_fused_add_native_layer_norm_169(c_void_p(buf763.data_ptr()), c_void_p(buf783.data_ptr()), c_void_p(arg447_1.data_ptr()), c_void_p(arg448_1.data_ptr()), c_void_p(buf784.data_ptr()), c_void_p(buf785.data_ptr()), c_void_p(buf787.data_ptr()))
    del arg447_1
    del arg448_1
    buf788 = reinterpret_tensor(buf743, (128, 10240), (10240, 1), 0); del buf743  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_15_fc1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg450_1, reinterpret_tensor(buf787, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg449_1, (2560, 10240), (1, 2560), 0), alpha=1, beta=1, out=buf788)
    del arg449_1
    del arg450_1
    buf789 = reinterpret_tensor(buf788, (1, 128, 10240), (1310720, 10240, 1), 0); del buf788  # reuse
    cpp_fused_gelu_170(c_void_p(buf789.data_ptr()))
    buf790 = reinterpret_tensor(buf787, (128, 2560), (2560, 1), 0); del buf787  # reuse
    # Source Nodes: [hidden_states_263], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg452_1, reinterpret_tensor(buf789, (128, 10240), (10240, 1), 0), reinterpret_tensor(arg451_1, (10240, 2560), (1, 10240), 0), alpha=1, beta=1, out=buf790)
    del arg451_1
    del arg452_1
    buf791 = buf785; del buf785  # reuse
    buf792 = buf784; del buf784  # reuse
    buf794 = reinterpret_tensor(buf772, (1, 128, 2560), (327680, 2560, 1), 0); del buf772  # reuse
    cpp_fused_add_native_layer_norm_171(c_void_p(buf763.data_ptr()), c_void_p(buf783.data_ptr()), c_void_p(buf790.data_ptr()), c_void_p(arg453_1.data_ptr()), c_void_p(arg454_1.data_ptr()), c_void_p(buf791.data_ptr()), c_void_p(buf792.data_ptr()), c_void_p(buf794.data_ptr()))
    del arg453_1
    del arg454_1
    buf795 = reinterpret_tensor(buf771, (128, 2560), (2560, 1), 0); del buf771  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_16_self_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg456_1, reinterpret_tensor(buf794, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg455_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf795)
    del arg455_1
    del arg456_1
    buf796 = buf770; del buf770  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_16_self_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg458_1, reinterpret_tensor(buf794, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg457_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf796)
    del arg457_1
    del arg458_1
    buf797 = reinterpret_tensor(buf769, (1, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf769  # reuse
    buf798 = reinterpret_tensor(buf768, (1, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf768  # reuse
    cpp_fused_clone_172(c_void_p(buf795.data_ptr()), c_void_p(buf796.data_ptr()), c_void_p(buf797.data_ptr()), c_void_p(buf798.data_ptr()))
    buf799 = buf758; del buf758  # reuse
    # Source Nodes: [attn_weights_100], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf797, (32, 128, 80), (10240, 80, 1), 0), reinterpret_tensor(buf798, (32, 80, 128), (10240, 1, 80), 0), out=buf799)
    buf800 = buf756; del buf756  # reuse
    buf801 = buf799; del buf799  # reuse
    buf802 = buf754; del buf754  # reuse
    cpp_fused__softmax_173(c_void_p(buf801.data_ptr()), c_void_p(buf800.data_ptr()), c_void_p(buf802.data_ptr()))
    buf803 = reinterpret_tensor(buf798, (128, 2560), (2560, 1), 0); del buf798  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_16_self_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg460_1, reinterpret_tensor(buf794, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg459_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf803)
    del arg459_1
    del arg460_1
    buf804 = buf801; del buf801  # reuse
    buf805 = reinterpret_tensor(buf794, (1, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf794  # reuse
    cpp_fused__softmax_clone_174(c_void_p(buf804.data_ptr()), c_void_p(buf802.data_ptr()), c_void_p(buf803.data_ptr()), c_void_p(buf805.data_ptr()))
    buf806 = reinterpret_tensor(buf803, (32, 128, 80), (10240, 80, 1), 0); del buf803  # reuse
    # Source Nodes: [attn_output_170, attn_weights_103], Original ATen: [aten._softmax, aten.bmm]
    extern_kernels.bmm(buf804, reinterpret_tensor(buf805, (32, 128, 80), (10240, 80, 1), 0), out=buf806)
    buf807 = reinterpret_tensor(buf805, (1, 128, 32, 80), (327680, 2560, 80, 1), 0); del buf805  # reuse
    cpp_fused_clone_175(c_void_p(buf806.data_ptr()), c_void_p(buf807.data_ptr()))
    buf808 = reinterpret_tensor(buf806, (128, 2560), (2560, 1), 0); del buf806  # reuse
    # Source Nodes: [hidden_states_268], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg462_1, reinterpret_tensor(buf807, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg461_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf808)
    del arg461_1
    del arg462_1
    buf809 = buf792; del buf792  # reuse
    buf810 = buf791; del buf791  # reuse
    buf812 = reinterpret_tensor(buf807, (1, 128, 2560), (327680, 2560, 1), 0); del buf807  # reuse
    cpp_fused_add_native_layer_norm_176(c_void_p(buf763.data_ptr()), c_void_p(buf783.data_ptr()), c_void_p(buf790.data_ptr()), c_void_p(buf808.data_ptr()), c_void_p(arg463_1.data_ptr()), c_void_p(arg464_1.data_ptr()), c_void_p(buf809.data_ptr()), c_void_p(buf810.data_ptr()), c_void_p(buf812.data_ptr()))
    del arg463_1
    del arg464_1
    buf813 = reinterpret_tensor(buf797, (128, 2560), (2560, 1), 0); del buf797  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_16_encoder_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg466_1, reinterpret_tensor(buf812, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg465_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf813)
    del arg465_1
    del arg466_1
    buf814 = reinterpret_tensor(buf812, (128, 2560), (2560, 1), 0); del buf812  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_16_encoder_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg468_1, reinterpret_tensor(buf81, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg467_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf814)
    del arg467_1
    del arg468_1
    buf815 = buf796; del buf796  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_16_encoder_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg470_1, reinterpret_tensor(buf81, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg469_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf815)
    del arg469_1
    del arg470_1
    buf816 = reinterpret_tensor(buf795, (1, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf795  # reuse
    buf817 = reinterpret_tensor(buf750, (1, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf750  # reuse
    buf818 = reinterpret_tensor(buf749, (1, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf749  # reuse
    cpp_fused_clone_177(c_void_p(buf813.data_ptr()), c_void_p(buf814.data_ptr()), c_void_p(buf815.data_ptr()), c_void_p(buf816.data_ptr()), c_void_p(buf817.data_ptr()), c_void_p(buf818.data_ptr()))
    del buf813
    del buf814
    # Source Nodes: [], Original ATen: []
    buf819 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf816, (1, 32, 128, 80), (0, 10240, 80, 1), 0), reinterpret_tensor(buf817, (1, 32, 128, 80), (0, 10240, 80, 1), 0), reinterpret_tensor(buf818, (1, 32, 128, 80), (0, 10240, 80, 1), 0), scale=1.0)
    buf820 = buf819[0]
    del buf819
    buf827 = reinterpret_tensor(buf820, (1, 128, 32, 80), (327680, 2560, 80, 1), 0); del buf820  # reuse
    cpp_fused_clone_178(c_void_p(buf827.data_ptr()))
    buf828 = reinterpret_tensor(buf818, (128, 2560), (2560, 1), 0); del buf818  # reuse
    # Source Nodes: [hidden_states_272], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg472_1, reinterpret_tensor(buf827, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg471_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf828)
    del arg471_1
    del arg472_1
    buf829 = reinterpret_tensor(buf828, (1, 128, 2560), (327680, 2560, 1), 0); del buf828  # reuse
    buf830 = buf810; del buf810  # reuse
    buf831 = buf809; del buf809  # reuse
    buf833 = reinterpret_tensor(buf827, (1, 128, 2560), (327680, 2560, 1), 0); del buf827  # reuse
    cpp_fused_add_native_layer_norm_179(c_void_p(buf829.data_ptr()), c_void_p(buf763.data_ptr()), c_void_p(buf783.data_ptr()), c_void_p(buf790.data_ptr()), c_void_p(buf808.data_ptr()), c_void_p(arg473_1.data_ptr()), c_void_p(arg474_1.data_ptr()), c_void_p(buf830.data_ptr()), c_void_p(buf831.data_ptr()), c_void_p(buf833.data_ptr()))
    del arg473_1
    del arg474_1
    buf834 = reinterpret_tensor(buf789, (128, 10240), (10240, 1), 0); del buf789  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_16_fc1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg476_1, reinterpret_tensor(buf833, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg475_1, (2560, 10240), (1, 2560), 0), alpha=1, beta=1, out=buf834)
    del arg475_1
    del arg476_1
    buf835 = reinterpret_tensor(buf834, (1, 128, 10240), (1310720, 10240, 1), 0); del buf834  # reuse
    cpp_fused_gelu_180(c_void_p(buf835.data_ptr()))
    buf836 = reinterpret_tensor(buf833, (128, 2560), (2560, 1), 0); del buf833  # reuse
    # Source Nodes: [hidden_states_278], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg478_1, reinterpret_tensor(buf835, (128, 10240), (10240, 1), 0), reinterpret_tensor(arg477_1, (10240, 2560), (1, 10240), 0), alpha=1, beta=1, out=buf836)
    del arg477_1
    del arg478_1
    buf837 = buf831; del buf831  # reuse
    buf838 = buf830; del buf830  # reuse
    buf840 = reinterpret_tensor(buf808, (1, 128, 2560), (327680, 2560, 1), 0); del buf808  # reuse
    cpp_fused_add_native_layer_norm_181(c_void_p(buf829.data_ptr()), c_void_p(buf836.data_ptr()), c_void_p(arg479_1.data_ptr()), c_void_p(arg480_1.data_ptr()), c_void_p(buf837.data_ptr()), c_void_p(buf838.data_ptr()), c_void_p(buf840.data_ptr()))
    del arg479_1
    del arg480_1
    buf841 = buf790; del buf790  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_17_self_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg482_1, reinterpret_tensor(buf840, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg481_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf841)
    del arg481_1
    del arg482_1
    buf842 = buf783; del buf783  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_17_self_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg484_1, reinterpret_tensor(buf840, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg483_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf842)
    del arg483_1
    del arg484_1
    buf843 = reinterpret_tensor(buf763, (1, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf763  # reuse
    buf844 = buf817; del buf817  # reuse
    cpp_fused_clone_182(c_void_p(buf841.data_ptr()), c_void_p(buf842.data_ptr()), c_void_p(buf843.data_ptr()), c_void_p(buf844.data_ptr()))
    buf845 = buf804; del buf804  # reuse
    # Source Nodes: [attn_weights_106], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf843, (32, 128, 80), (10240, 80, 1), 0), reinterpret_tensor(buf844, (32, 80, 128), (10240, 1, 80), 0), out=buf845)
    buf846 = buf802; del buf802  # reuse
    buf847 = buf845; del buf845  # reuse
    buf848 = buf800; del buf800  # reuse
    cpp_fused__softmax_183(c_void_p(buf847.data_ptr()), c_void_p(buf846.data_ptr()), c_void_p(buf848.data_ptr()))
    buf849 = reinterpret_tensor(buf844, (128, 2560), (2560, 1), 0); del buf844  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_17_self_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg486_1, reinterpret_tensor(buf840, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg485_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf849)
    del arg485_1
    del arg486_1
    buf850 = buf847; del buf847  # reuse
    buf851 = reinterpret_tensor(buf840, (1, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf840  # reuse
    cpp_fused__softmax_clone_184(c_void_p(buf850.data_ptr()), c_void_p(buf848.data_ptr()), c_void_p(buf849.data_ptr()), c_void_p(buf851.data_ptr()))
    buf852 = reinterpret_tensor(buf849, (32, 128, 80), (10240, 80, 1), 0); del buf849  # reuse
    # Source Nodes: [attn_output_180, attn_weights_109], Original ATen: [aten._softmax, aten.bmm]
    extern_kernels.bmm(buf850, reinterpret_tensor(buf851, (32, 128, 80), (10240, 80, 1), 0), out=buf852)
    buf853 = reinterpret_tensor(buf851, (1, 128, 32, 80), (327680, 2560, 80, 1), 0); del buf851  # reuse
    cpp_fused_clone_185(c_void_p(buf852.data_ptr()), c_void_p(buf853.data_ptr()))
    buf854 = reinterpret_tensor(buf852, (128, 2560), (2560, 1), 0); del buf852  # reuse
    # Source Nodes: [hidden_states_283], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg488_1, reinterpret_tensor(buf853, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg487_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf854)
    del arg487_1
    del arg488_1
    buf855 = buf838; del buf838  # reuse
    buf856 = buf837; del buf837  # reuse
    buf858 = reinterpret_tensor(buf853, (1, 128, 2560), (327680, 2560, 1), 0); del buf853  # reuse
    cpp_fused_add_native_layer_norm_186(c_void_p(buf829.data_ptr()), c_void_p(buf836.data_ptr()), c_void_p(buf854.data_ptr()), c_void_p(arg489_1.data_ptr()), c_void_p(arg490_1.data_ptr()), c_void_p(buf855.data_ptr()), c_void_p(buf856.data_ptr()), c_void_p(buf858.data_ptr()))
    del arg489_1
    del arg490_1
    buf859 = reinterpret_tensor(buf843, (128, 2560), (2560, 1), 0); del buf843  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_17_encoder_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg492_1, reinterpret_tensor(buf858, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg491_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf859)
    del arg491_1
    del arg492_1
    buf860 = reinterpret_tensor(buf858, (128, 2560), (2560, 1), 0); del buf858  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_17_encoder_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg494_1, reinterpret_tensor(buf81, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg493_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf860)
    del arg493_1
    del arg494_1
    buf861 = buf842; del buf842  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_17_encoder_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg496_1, reinterpret_tensor(buf81, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg495_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf861)
    del arg495_1
    del arg496_1
    buf862 = reinterpret_tensor(buf841, (1, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf841  # reuse
    buf863 = buf816; del buf816  # reuse
    buf864 = reinterpret_tensor(buf815, (1, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf815  # reuse
    cpp_fused_clone_187(c_void_p(buf859.data_ptr()), c_void_p(buf860.data_ptr()), c_void_p(buf861.data_ptr()), c_void_p(buf862.data_ptr()), c_void_p(buf863.data_ptr()), c_void_p(buf864.data_ptr()))
    del buf859
    del buf860
    # Source Nodes: [], Original ATen: []
    buf865 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf862, (1, 32, 128, 80), (0, 10240, 80, 1), 0), reinterpret_tensor(buf863, (1, 32, 128, 80), (0, 10240, 80, 1), 0), reinterpret_tensor(buf864, (1, 32, 128, 80), (0, 10240, 80, 1), 0), scale=1.0)
    buf866 = buf865[0]
    del buf865
    buf873 = reinterpret_tensor(buf866, (1, 128, 32, 80), (327680, 2560, 80, 1), 0); del buf866  # reuse
    cpp_fused_clone_188(c_void_p(buf873.data_ptr()))
    buf874 = reinterpret_tensor(buf864, (128, 2560), (2560, 1), 0); del buf864  # reuse
    # Source Nodes: [hidden_states_287], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg498_1, reinterpret_tensor(buf873, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg497_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf874)
    del arg497_1
    del arg498_1
    buf875 = buf856; del buf856  # reuse
    buf876 = buf855; del buf855  # reuse
    buf878 = reinterpret_tensor(buf873, (1, 128, 2560), (327680, 2560, 1), 0); del buf873  # reuse
    cpp_fused_add_native_layer_norm_189(c_void_p(buf829.data_ptr()), c_void_p(buf836.data_ptr()), c_void_p(buf854.data_ptr()), c_void_p(buf874.data_ptr()), c_void_p(arg499_1.data_ptr()), c_void_p(arg500_1.data_ptr()), c_void_p(buf875.data_ptr()), c_void_p(buf876.data_ptr()), c_void_p(buf878.data_ptr()))
    del arg499_1
    del arg500_1
    buf879 = reinterpret_tensor(buf835, (128, 10240), (10240, 1), 0); del buf835  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_17_fc1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg502_1, reinterpret_tensor(buf878, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg501_1, (2560, 10240), (1, 2560), 0), alpha=1, beta=1, out=buf879)
    del arg501_1
    del arg502_1
    buf880 = reinterpret_tensor(buf879, (1, 128, 10240), (1310720, 10240, 1), 0); del buf879  # reuse
    cpp_fused_gelu_190(c_void_p(buf880.data_ptr()))
    buf881 = reinterpret_tensor(buf878, (128, 2560), (2560, 1), 0); del buf878  # reuse
    # Source Nodes: [hidden_states_293], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg504_1, reinterpret_tensor(buf880, (128, 10240), (10240, 1), 0), reinterpret_tensor(arg503_1, (10240, 2560), (1, 10240), 0), alpha=1, beta=1, out=buf881)
    del arg503_1
    del arg504_1
    buf882 = reinterpret_tensor(buf881, (1, 128, 2560), (327680, 2560, 1), 0); del buf881  # reuse
    buf883 = buf876; del buf876  # reuse
    buf884 = buf875; del buf875  # reuse
    buf886 = reinterpret_tensor(buf863, (1, 128, 2560), (327680, 2560, 1), 0); del buf863  # reuse
    cpp_fused_add_native_layer_norm_191(c_void_p(buf882.data_ptr()), c_void_p(buf829.data_ptr()), c_void_p(buf836.data_ptr()), c_void_p(buf854.data_ptr()), c_void_p(buf874.data_ptr()), c_void_p(arg505_1.data_ptr()), c_void_p(arg506_1.data_ptr()), c_void_p(buf883.data_ptr()), c_void_p(buf884.data_ptr()), c_void_p(buf886.data_ptr()))
    del arg505_1
    del arg506_1
    buf887 = buf874; del buf874  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_18_self_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg508_1, reinterpret_tensor(buf886, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg507_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf887)
    del arg507_1
    del arg508_1
    buf888 = buf854; del buf854  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_18_self_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg510_1, reinterpret_tensor(buf886, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg509_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf888)
    del arg509_1
    del arg510_1
    buf889 = reinterpret_tensor(buf836, (1, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf836  # reuse
    buf890 = reinterpret_tensor(buf829, (1, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf829  # reuse
    cpp_fused_clone_192(c_void_p(buf887.data_ptr()), c_void_p(buf888.data_ptr()), c_void_p(buf889.data_ptr()), c_void_p(buf890.data_ptr()))
    buf891 = buf850; del buf850  # reuse
    # Source Nodes: [attn_weights_112], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf889, (32, 128, 80), (10240, 80, 1), 0), reinterpret_tensor(buf890, (32, 80, 128), (10240, 1, 80), 0), out=buf891)
    buf892 = buf848; del buf848  # reuse
    buf893 = buf891; del buf891  # reuse
    buf894 = buf846; del buf846  # reuse
    cpp_fused__softmax_193(c_void_p(buf893.data_ptr()), c_void_p(buf892.data_ptr()), c_void_p(buf894.data_ptr()))
    buf895 = reinterpret_tensor(buf890, (128, 2560), (2560, 1), 0); del buf890  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_18_self_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg512_1, reinterpret_tensor(buf886, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg511_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf895)
    del arg511_1
    del arg512_1
    buf896 = buf893; del buf893  # reuse
    buf897 = reinterpret_tensor(buf886, (1, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf886  # reuse
    cpp_fused__softmax_clone_194(c_void_p(buf896.data_ptr()), c_void_p(buf894.data_ptr()), c_void_p(buf895.data_ptr()), c_void_p(buf897.data_ptr()))
    buf898 = reinterpret_tensor(buf895, (32, 128, 80), (10240, 80, 1), 0); del buf895  # reuse
    # Source Nodes: [attn_output_190, attn_weights_115], Original ATen: [aten._softmax, aten.bmm]
    extern_kernels.bmm(buf896, reinterpret_tensor(buf897, (32, 128, 80), (10240, 80, 1), 0), out=buf898)
    buf899 = reinterpret_tensor(buf897, (1, 128, 32, 80), (327680, 2560, 80, 1), 0); del buf897  # reuse
    cpp_fused_clone_195(c_void_p(buf898.data_ptr()), c_void_p(buf899.data_ptr()))
    buf900 = reinterpret_tensor(buf898, (128, 2560), (2560, 1), 0); del buf898  # reuse
    # Source Nodes: [hidden_states_298], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg514_1, reinterpret_tensor(buf899, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg513_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf900)
    del arg513_1
    del arg514_1
    buf901 = buf884; del buf884  # reuse
    buf902 = buf883; del buf883  # reuse
    buf904 = reinterpret_tensor(buf899, (1, 128, 2560), (327680, 2560, 1), 0); del buf899  # reuse
    cpp_fused_add_native_layer_norm_196(c_void_p(buf882.data_ptr()), c_void_p(buf900.data_ptr()), c_void_p(arg515_1.data_ptr()), c_void_p(arg516_1.data_ptr()), c_void_p(buf901.data_ptr()), c_void_p(buf902.data_ptr()), c_void_p(buf904.data_ptr()))
    del arg515_1
    del arg516_1
    buf905 = reinterpret_tensor(buf889, (128, 2560), (2560, 1), 0); del buf889  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_18_encoder_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg518_1, reinterpret_tensor(buf904, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg517_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf905)
    del arg517_1
    del arg518_1
    buf906 = reinterpret_tensor(buf904, (128, 2560), (2560, 1), 0); del buf904  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_18_encoder_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg520_1, reinterpret_tensor(buf81, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg519_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf906)
    del arg519_1
    del arg520_1
    buf907 = buf888; del buf888  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_18_encoder_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg522_1, reinterpret_tensor(buf81, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg521_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf907)
    del arg521_1
    del arg522_1
    buf908 = reinterpret_tensor(buf887, (1, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf887  # reuse
    buf909 = buf862; del buf862  # reuse
    buf910 = reinterpret_tensor(buf861, (1, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf861  # reuse
    cpp_fused_clone_197(c_void_p(buf905.data_ptr()), c_void_p(buf906.data_ptr()), c_void_p(buf907.data_ptr()), c_void_p(buf908.data_ptr()), c_void_p(buf909.data_ptr()), c_void_p(buf910.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf911 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf908, (1, 32, 128, 80), (0, 10240, 80, 1), 0), reinterpret_tensor(buf909, (1, 32, 128, 80), (0, 10240, 80, 1), 0), reinterpret_tensor(buf910, (1, 32, 128, 80), (0, 10240, 80, 1), 0), scale=1.0)
    buf912 = buf911[0]
    del buf911
    buf919 = reinterpret_tensor(buf912, (1, 128, 32, 80), (327680, 2560, 80, 1), 0); del buf912  # reuse
    cpp_fused_clone_198(c_void_p(buf919.data_ptr()))
    buf920 = reinterpret_tensor(buf910, (128, 2560), (2560, 1), 0); del buf910  # reuse
    # Source Nodes: [hidden_states_302], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg524_1, reinterpret_tensor(buf919, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg523_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf920)
    del arg523_1
    del arg524_1
    buf921 = buf902; del buf902  # reuse
    buf922 = buf901; del buf901  # reuse
    buf924 = reinterpret_tensor(buf919, (1, 128, 2560), (327680, 2560, 1), 0); del buf919  # reuse
    cpp_fused_add_native_layer_norm_199(c_void_p(buf882.data_ptr()), c_void_p(buf900.data_ptr()), c_void_p(buf920.data_ptr()), c_void_p(arg525_1.data_ptr()), c_void_p(arg526_1.data_ptr()), c_void_p(buf921.data_ptr()), c_void_p(buf922.data_ptr()), c_void_p(buf924.data_ptr()))
    del arg525_1
    del arg526_1
    buf925 = reinterpret_tensor(buf880, (128, 10240), (10240, 1), 0); del buf880  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_18_fc1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg528_1, reinterpret_tensor(buf924, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg527_1, (2560, 10240), (1, 2560), 0), alpha=1, beta=1, out=buf925)
    del arg527_1
    del arg528_1
    buf926 = reinterpret_tensor(buf925, (1, 128, 10240), (1310720, 10240, 1), 0); del buf925  # reuse
    cpp_fused_gelu_200(c_void_p(buf926.data_ptr()))
    buf927 = reinterpret_tensor(buf924, (128, 2560), (2560, 1), 0); del buf924  # reuse
    # Source Nodes: [hidden_states_308], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg530_1, reinterpret_tensor(buf926, (128, 10240), (10240, 1), 0), reinterpret_tensor(arg529_1, (10240, 2560), (1, 10240), 0), alpha=1, beta=1, out=buf927)
    del arg529_1
    del arg530_1
    buf928 = buf922; del buf922  # reuse
    buf929 = buf921; del buf921  # reuse
    buf931 = reinterpret_tensor(buf909, (1, 128, 2560), (327680, 2560, 1), 0); del buf909  # reuse
    cpp_fused_add_native_layer_norm_201(c_void_p(buf882.data_ptr()), c_void_p(buf900.data_ptr()), c_void_p(buf920.data_ptr()), c_void_p(buf927.data_ptr()), c_void_p(arg531_1.data_ptr()), c_void_p(arg532_1.data_ptr()), c_void_p(buf928.data_ptr()), c_void_p(buf929.data_ptr()), c_void_p(buf931.data_ptr()))
    del arg531_1
    del arg532_1
    buf932 = reinterpret_tensor(buf908, (128, 2560), (2560, 1), 0); del buf908  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_19_self_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg534_1, reinterpret_tensor(buf931, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg533_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf932)
    del arg533_1
    del arg534_1
    buf933 = buf907; del buf907  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_19_self_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg536_1, reinterpret_tensor(buf931, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg535_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf933)
    del arg535_1
    del arg536_1
    buf934 = reinterpret_tensor(buf906, (1, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf906  # reuse
    buf935 = reinterpret_tensor(buf905, (1, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf905  # reuse
    cpp_fused_clone_202(c_void_p(buf932.data_ptr()), c_void_p(buf933.data_ptr()), c_void_p(buf934.data_ptr()), c_void_p(buf935.data_ptr()))
    buf936 = buf896; del buf896  # reuse
    # Source Nodes: [attn_weights_118], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf934, (32, 128, 80), (10240, 80, 1), 0), reinterpret_tensor(buf935, (32, 80, 128), (10240, 1, 80), 0), out=buf936)
    buf937 = buf894; del buf894  # reuse
    buf938 = buf936; del buf936  # reuse
    buf939 = buf892; del buf892  # reuse
    cpp_fused__softmax_203(c_void_p(buf938.data_ptr()), c_void_p(buf937.data_ptr()), c_void_p(buf939.data_ptr()))
    buf940 = reinterpret_tensor(buf935, (128, 2560), (2560, 1), 0); del buf935  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_19_self_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg538_1, reinterpret_tensor(buf931, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg537_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf940)
    del arg537_1
    del arg538_1
    buf941 = buf938; del buf938  # reuse
    buf942 = reinterpret_tensor(buf931, (1, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf931  # reuse
    cpp_fused__softmax_clone_204(c_void_p(buf941.data_ptr()), c_void_p(buf939.data_ptr()), c_void_p(buf940.data_ptr()), c_void_p(buf942.data_ptr()))
    buf943 = reinterpret_tensor(buf940, (32, 128, 80), (10240, 80, 1), 0); del buf940  # reuse
    # Source Nodes: [attn_output_200, attn_weights_121], Original ATen: [aten._softmax, aten.bmm]
    extern_kernels.bmm(buf941, reinterpret_tensor(buf942, (32, 128, 80), (10240, 80, 1), 0), out=buf943)
    buf944 = reinterpret_tensor(buf942, (1, 128, 32, 80), (327680, 2560, 80, 1), 0); del buf942  # reuse
    cpp_fused_clone_205(c_void_p(buf943.data_ptr()), c_void_p(buf944.data_ptr()))
    buf945 = reinterpret_tensor(buf943, (128, 2560), (2560, 1), 0); del buf943  # reuse
    # Source Nodes: [hidden_states_313], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg540_1, reinterpret_tensor(buf944, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg539_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf945)
    del arg539_1
    del arg540_1
    buf946 = reinterpret_tensor(buf945, (1, 128, 2560), (327680, 2560, 1), 0); del buf945  # reuse
    buf947 = buf929; del buf929  # reuse
    buf948 = buf928; del buf928  # reuse
    buf950 = reinterpret_tensor(buf944, (1, 128, 2560), (327680, 2560, 1), 0); del buf944  # reuse
    cpp_fused_add_native_layer_norm_206(c_void_p(buf946.data_ptr()), c_void_p(buf882.data_ptr()), c_void_p(buf900.data_ptr()), c_void_p(buf920.data_ptr()), c_void_p(buf927.data_ptr()), c_void_p(arg541_1.data_ptr()), c_void_p(arg542_1.data_ptr()), c_void_p(buf947.data_ptr()), c_void_p(buf948.data_ptr()), c_void_p(buf950.data_ptr()))
    del arg541_1
    del arg542_1
    buf951 = buf927; del buf927  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_19_encoder_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg544_1, reinterpret_tensor(buf950, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg543_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf951)
    del arg543_1
    del arg544_1
    buf952 = reinterpret_tensor(buf950, (128, 2560), (2560, 1), 0); del buf950  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_19_encoder_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg546_1, reinterpret_tensor(buf81, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg545_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf952)
    del arg545_1
    del arg546_1
    buf953 = buf920; del buf920  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_19_encoder_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg548_1, reinterpret_tensor(buf81, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg547_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf953)
    del arg547_1
    del arg548_1
    buf954 = reinterpret_tensor(buf900, (1, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf900  # reuse
    buf955 = reinterpret_tensor(buf882, (1, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf882  # reuse
    buf956 = buf934; del buf934  # reuse
    cpp_fused_clone_207(c_void_p(buf951.data_ptr()), c_void_p(buf952.data_ptr()), c_void_p(buf953.data_ptr()), c_void_p(buf954.data_ptr()), c_void_p(buf955.data_ptr()), c_void_p(buf956.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf957 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf954, (1, 32, 128, 80), (0, 10240, 80, 1), 0), reinterpret_tensor(buf955, (1, 32, 128, 80), (0, 10240, 80, 1), 0), reinterpret_tensor(buf956, (1, 32, 128, 80), (0, 10240, 80, 1), 0), scale=1.0)
    buf958 = buf957[0]
    del buf957
    buf965 = reinterpret_tensor(buf958, (1, 128, 32, 80), (327680, 2560, 80, 1), 0); del buf958  # reuse
    cpp_fused_clone_208(c_void_p(buf965.data_ptr()))
    buf966 = reinterpret_tensor(buf956, (128, 2560), (2560, 1), 0); del buf956  # reuse
    # Source Nodes: [hidden_states_317], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg550_1, reinterpret_tensor(buf965, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg549_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf966)
    del arg549_1
    del arg550_1
    buf967 = buf948; del buf948  # reuse
    buf968 = buf947; del buf947  # reuse
    buf970 = reinterpret_tensor(buf965, (1, 128, 2560), (327680, 2560, 1), 0); del buf965  # reuse
    cpp_fused_add_native_layer_norm_209(c_void_p(buf946.data_ptr()), c_void_p(buf966.data_ptr()), c_void_p(arg551_1.data_ptr()), c_void_p(arg552_1.data_ptr()), c_void_p(buf967.data_ptr()), c_void_p(buf968.data_ptr()), c_void_p(buf970.data_ptr()))
    del arg551_1
    del arg552_1
    buf971 = reinterpret_tensor(buf926, (128, 10240), (10240, 1), 0); del buf926  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_19_fc1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg554_1, reinterpret_tensor(buf970, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg553_1, (2560, 10240), (1, 2560), 0), alpha=1, beta=1, out=buf971)
    del arg553_1
    del arg554_1
    buf972 = reinterpret_tensor(buf971, (1, 128, 10240), (1310720, 10240, 1), 0); del buf971  # reuse
    cpp_fused_gelu_210(c_void_p(buf972.data_ptr()))
    buf973 = reinterpret_tensor(buf970, (128, 2560), (2560, 1), 0); del buf970  # reuse
    # Source Nodes: [hidden_states_323], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg556_1, reinterpret_tensor(buf972, (128, 10240), (10240, 1), 0), reinterpret_tensor(arg555_1, (10240, 2560), (1, 10240), 0), alpha=1, beta=1, out=buf973)
    del arg555_1
    del arg556_1
    buf974 = buf968; del buf968  # reuse
    buf975 = buf967; del buf967  # reuse
    buf977 = reinterpret_tensor(buf955, (1, 128, 2560), (327680, 2560, 1), 0); del buf955  # reuse
    cpp_fused_add_native_layer_norm_211(c_void_p(buf946.data_ptr()), c_void_p(buf966.data_ptr()), c_void_p(buf973.data_ptr()), c_void_p(arg557_1.data_ptr()), c_void_p(arg558_1.data_ptr()), c_void_p(buf974.data_ptr()), c_void_p(buf975.data_ptr()), c_void_p(buf977.data_ptr()))
    del arg557_1
    del arg558_1
    buf978 = reinterpret_tensor(buf954, (128, 2560), (2560, 1), 0); del buf954  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_20_self_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg560_1, reinterpret_tensor(buf977, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg559_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf978)
    del arg559_1
    del arg560_1
    buf979 = buf953; del buf953  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_20_self_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg562_1, reinterpret_tensor(buf977, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg561_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf979)
    del arg561_1
    del arg562_1
    buf980 = reinterpret_tensor(buf952, (1, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf952  # reuse
    buf981 = reinterpret_tensor(buf951, (1, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf951  # reuse
    cpp_fused_clone_212(c_void_p(buf978.data_ptr()), c_void_p(buf979.data_ptr()), c_void_p(buf980.data_ptr()), c_void_p(buf981.data_ptr()))
    buf982 = buf941; del buf941  # reuse
    # Source Nodes: [attn_weights_124], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf980, (32, 128, 80), (10240, 80, 1), 0), reinterpret_tensor(buf981, (32, 80, 128), (10240, 1, 80), 0), out=buf982)
    buf983 = buf939; del buf939  # reuse
    buf984 = buf982; del buf982  # reuse
    buf985 = buf937; del buf937  # reuse
    cpp_fused__softmax_213(c_void_p(buf984.data_ptr()), c_void_p(buf983.data_ptr()), c_void_p(buf985.data_ptr()))
    buf986 = reinterpret_tensor(buf981, (128, 2560), (2560, 1), 0); del buf981  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_20_self_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg564_1, reinterpret_tensor(buf977, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg563_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf986)
    del arg563_1
    del arg564_1
    buf987 = buf984; del buf984  # reuse
    buf988 = reinterpret_tensor(buf977, (1, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf977  # reuse
    cpp_fused__softmax_clone_214(c_void_p(buf987.data_ptr()), c_void_p(buf985.data_ptr()), c_void_p(buf986.data_ptr()), c_void_p(buf988.data_ptr()))
    buf989 = reinterpret_tensor(buf986, (32, 128, 80), (10240, 80, 1), 0); del buf986  # reuse
    # Source Nodes: [attn_output_210, attn_weights_127], Original ATen: [aten._softmax, aten.bmm]
    extern_kernels.bmm(buf987, reinterpret_tensor(buf988, (32, 128, 80), (10240, 80, 1), 0), out=buf989)
    buf990 = reinterpret_tensor(buf988, (1, 128, 32, 80), (327680, 2560, 80, 1), 0); del buf988  # reuse
    cpp_fused_clone_215(c_void_p(buf989.data_ptr()), c_void_p(buf990.data_ptr()))
    buf991 = reinterpret_tensor(buf989, (128, 2560), (2560, 1), 0); del buf989  # reuse
    # Source Nodes: [hidden_states_328], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg566_1, reinterpret_tensor(buf990, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg565_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf991)
    del arg565_1
    del arg566_1
    buf992 = buf975; del buf975  # reuse
    buf993 = buf974; del buf974  # reuse
    buf995 = reinterpret_tensor(buf990, (1, 128, 2560), (327680, 2560, 1), 0); del buf990  # reuse
    cpp_fused_add_native_layer_norm_216(c_void_p(buf946.data_ptr()), c_void_p(buf966.data_ptr()), c_void_p(buf973.data_ptr()), c_void_p(buf991.data_ptr()), c_void_p(arg567_1.data_ptr()), c_void_p(arg568_1.data_ptr()), c_void_p(buf992.data_ptr()), c_void_p(buf993.data_ptr()), c_void_p(buf995.data_ptr()))
    del arg567_1
    del arg568_1
    buf996 = reinterpret_tensor(buf980, (128, 2560), (2560, 1), 0); del buf980  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_20_encoder_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg570_1, reinterpret_tensor(buf995, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg569_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf996)
    del arg569_1
    del arg570_1
    buf997 = reinterpret_tensor(buf995, (128, 2560), (2560, 1), 0); del buf995  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_20_encoder_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg572_1, reinterpret_tensor(buf81, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg571_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf997)
    del arg571_1
    del arg572_1
    buf998 = buf979; del buf979  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_20_encoder_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg574_1, reinterpret_tensor(buf81, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg573_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf998)
    del arg573_1
    del arg574_1
    buf999 = reinterpret_tensor(buf978, (1, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf978  # reuse
    buf1000 = reinterpret_tensor(buf933, (1, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf933  # reuse
    buf1001 = reinterpret_tensor(buf932, (1, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf932  # reuse
    cpp_fused_clone_217(c_void_p(buf996.data_ptr()), c_void_p(buf997.data_ptr()), c_void_p(buf998.data_ptr()), c_void_p(buf999.data_ptr()), c_void_p(buf1000.data_ptr()), c_void_p(buf1001.data_ptr()))
    del buf996
    del buf997
    # Source Nodes: [], Original ATen: []
    buf1002 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf999, (1, 32, 128, 80), (0, 10240, 80, 1), 0), reinterpret_tensor(buf1000, (1, 32, 128, 80), (0, 10240, 80, 1), 0), reinterpret_tensor(buf1001, (1, 32, 128, 80), (0, 10240, 80, 1), 0), scale=1.0)
    buf1003 = buf1002[0]
    del buf1002
    buf1010 = reinterpret_tensor(buf1003, (1, 128, 32, 80), (327680, 2560, 80, 1), 0); del buf1003  # reuse
    cpp_fused_clone_218(c_void_p(buf1010.data_ptr()))
    buf1011 = reinterpret_tensor(buf999, (128, 2560), (2560, 1), 0); del buf999  # reuse
    # Source Nodes: [hidden_states_332], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg576_1, reinterpret_tensor(buf1010, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg575_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf1011)
    del arg575_1
    del arg576_1
    buf1012 = reinterpret_tensor(buf1011, (1, 128, 2560), (327680, 2560, 1), 0); del buf1011  # reuse
    buf1013 = buf993; del buf993  # reuse
    buf1014 = buf992; del buf992  # reuse
    buf1016 = reinterpret_tensor(buf1010, (1, 128, 2560), (327680, 2560, 1), 0); del buf1010  # reuse
    cpp_fused_add_native_layer_norm_219(c_void_p(buf1012.data_ptr()), c_void_p(buf946.data_ptr()), c_void_p(buf966.data_ptr()), c_void_p(buf973.data_ptr()), c_void_p(buf991.data_ptr()), c_void_p(arg577_1.data_ptr()), c_void_p(arg578_1.data_ptr()), c_void_p(buf1013.data_ptr()), c_void_p(buf1014.data_ptr()), c_void_p(buf1016.data_ptr()))
    del arg577_1
    del arg578_1
    buf1017 = reinterpret_tensor(buf972, (128, 10240), (10240, 1), 0); del buf972  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_20_fc1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg580_1, reinterpret_tensor(buf1016, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg579_1, (2560, 10240), (1, 2560), 0), alpha=1, beta=1, out=buf1017)
    del arg579_1
    del arg580_1
    buf1018 = reinterpret_tensor(buf1017, (1, 128, 10240), (1310720, 10240, 1), 0); del buf1017  # reuse
    cpp_fused_gelu_220(c_void_p(buf1018.data_ptr()))
    buf1019 = reinterpret_tensor(buf1016, (128, 2560), (2560, 1), 0); del buf1016  # reuse
    # Source Nodes: [hidden_states_338], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg582_1, reinterpret_tensor(buf1018, (128, 10240), (10240, 1), 0), reinterpret_tensor(arg581_1, (10240, 2560), (1, 10240), 0), alpha=1, beta=1, out=buf1019)
    del arg581_1
    del arg582_1
    buf1020 = buf1014; del buf1014  # reuse
    buf1021 = buf1013; del buf1013  # reuse
    buf1023 = reinterpret_tensor(buf991, (1, 128, 2560), (327680, 2560, 1), 0); del buf991  # reuse
    cpp_fused_add_native_layer_norm_221(c_void_p(buf1012.data_ptr()), c_void_p(buf1019.data_ptr()), c_void_p(arg583_1.data_ptr()), c_void_p(arg584_1.data_ptr()), c_void_p(buf1020.data_ptr()), c_void_p(buf1021.data_ptr()), c_void_p(buf1023.data_ptr()))
    del arg583_1
    del arg584_1
    buf1024 = buf973; del buf973  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_21_self_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg586_1, reinterpret_tensor(buf1023, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg585_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf1024)
    del arg585_1
    del arg586_1
    buf1025 = buf966; del buf966  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_21_self_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg588_1, reinterpret_tensor(buf1023, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg587_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf1025)
    del arg587_1
    del arg588_1
    buf1026 = reinterpret_tensor(buf946, (1, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf946  # reuse
    buf1027 = buf1001; del buf1001  # reuse
    cpp_fused_clone_222(c_void_p(buf1024.data_ptr()), c_void_p(buf1025.data_ptr()), c_void_p(buf1026.data_ptr()), c_void_p(buf1027.data_ptr()))
    buf1028 = buf987; del buf987  # reuse
    # Source Nodes: [attn_weights_130], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf1026, (32, 128, 80), (10240, 80, 1), 0), reinterpret_tensor(buf1027, (32, 80, 128), (10240, 1, 80), 0), out=buf1028)
    buf1029 = buf985; del buf985  # reuse
    buf1030 = buf1028; del buf1028  # reuse
    buf1031 = buf983; del buf983  # reuse
    cpp_fused__softmax_223(c_void_p(buf1030.data_ptr()), c_void_p(buf1029.data_ptr()), c_void_p(buf1031.data_ptr()))
    buf1032 = reinterpret_tensor(buf1027, (128, 2560), (2560, 1), 0); del buf1027  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_21_self_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg590_1, reinterpret_tensor(buf1023, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg589_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf1032)
    del arg589_1
    del arg590_1
    buf1033 = buf1030; del buf1030  # reuse
    buf1034 = reinterpret_tensor(buf1023, (1, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf1023  # reuse
    cpp_fused__softmax_clone_224(c_void_p(buf1033.data_ptr()), c_void_p(buf1031.data_ptr()), c_void_p(buf1032.data_ptr()), c_void_p(buf1034.data_ptr()))
    buf1035 = reinterpret_tensor(buf1032, (32, 128, 80), (10240, 80, 1), 0); del buf1032  # reuse
    # Source Nodes: [attn_output_220, attn_weights_133], Original ATen: [aten._softmax, aten.bmm]
    extern_kernels.bmm(buf1033, reinterpret_tensor(buf1034, (32, 128, 80), (10240, 80, 1), 0), out=buf1035)
    buf1036 = reinterpret_tensor(buf1034, (1, 128, 32, 80), (327680, 2560, 80, 1), 0); del buf1034  # reuse
    cpp_fused_clone_225(c_void_p(buf1035.data_ptr()), c_void_p(buf1036.data_ptr()))
    buf1037 = reinterpret_tensor(buf1035, (128, 2560), (2560, 1), 0); del buf1035  # reuse
    # Source Nodes: [hidden_states_343], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg592_1, reinterpret_tensor(buf1036, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg591_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf1037)
    del arg591_1
    del arg592_1
    buf1038 = buf1021; del buf1021  # reuse
    buf1039 = buf1020; del buf1020  # reuse
    buf1041 = reinterpret_tensor(buf1036, (1, 128, 2560), (327680, 2560, 1), 0); del buf1036  # reuse
    cpp_fused_add_native_layer_norm_226(c_void_p(buf1012.data_ptr()), c_void_p(buf1019.data_ptr()), c_void_p(buf1037.data_ptr()), c_void_p(arg593_1.data_ptr()), c_void_p(arg594_1.data_ptr()), c_void_p(buf1038.data_ptr()), c_void_p(buf1039.data_ptr()), c_void_p(buf1041.data_ptr()))
    del arg593_1
    del arg594_1
    buf1042 = reinterpret_tensor(buf1026, (128, 2560), (2560, 1), 0); del buf1026  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_21_encoder_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg596_1, reinterpret_tensor(buf1041, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg595_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf1042)
    del arg595_1
    del arg596_1
    buf1043 = reinterpret_tensor(buf1041, (128, 2560), (2560, 1), 0); del buf1041  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_21_encoder_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg598_1, reinterpret_tensor(buf81, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg597_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf1043)
    del arg597_1
    del arg598_1
    buf1044 = buf1025; del buf1025  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_21_encoder_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg600_1, reinterpret_tensor(buf81, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg599_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf1044)
    del arg599_1
    del arg600_1
    buf1045 = reinterpret_tensor(buf1024, (1, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf1024  # reuse
    buf1046 = buf1000; del buf1000  # reuse
    buf1047 = reinterpret_tensor(buf998, (1, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf998  # reuse
    cpp_fused_clone_227(c_void_p(buf1042.data_ptr()), c_void_p(buf1043.data_ptr()), c_void_p(buf1044.data_ptr()), c_void_p(buf1045.data_ptr()), c_void_p(buf1046.data_ptr()), c_void_p(buf1047.data_ptr()))
    del buf1042
    del buf1043
    # Source Nodes: [], Original ATen: []
    buf1048 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf1045, (1, 32, 128, 80), (0, 10240, 80, 1), 0), reinterpret_tensor(buf1046, (1, 32, 128, 80), (0, 10240, 80, 1), 0), reinterpret_tensor(buf1047, (1, 32, 128, 80), (0, 10240, 80, 1), 0), scale=1.0)
    buf1049 = buf1048[0]
    del buf1048
    buf1056 = reinterpret_tensor(buf1049, (1, 128, 32, 80), (327680, 2560, 80, 1), 0); del buf1049  # reuse
    cpp_fused_clone_228(c_void_p(buf1056.data_ptr()))
    buf1057 = reinterpret_tensor(buf1047, (128, 2560), (2560, 1), 0); del buf1047  # reuse
    # Source Nodes: [hidden_states_347], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg602_1, reinterpret_tensor(buf1056, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg601_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf1057)
    del arg601_1
    del arg602_1
    buf1058 = buf1039; del buf1039  # reuse
    buf1059 = buf1038; del buf1038  # reuse
    buf1061 = reinterpret_tensor(buf1056, (1, 128, 2560), (327680, 2560, 1), 0); del buf1056  # reuse
    cpp_fused_add_native_layer_norm_229(c_void_p(buf1012.data_ptr()), c_void_p(buf1019.data_ptr()), c_void_p(buf1037.data_ptr()), c_void_p(buf1057.data_ptr()), c_void_p(arg603_1.data_ptr()), c_void_p(arg604_1.data_ptr()), c_void_p(buf1058.data_ptr()), c_void_p(buf1059.data_ptr()), c_void_p(buf1061.data_ptr()))
    del arg603_1
    del arg604_1
    buf1062 = reinterpret_tensor(buf1018, (128, 10240), (10240, 1), 0); del buf1018  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_21_fc1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg606_1, reinterpret_tensor(buf1061, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg605_1, (2560, 10240), (1, 2560), 0), alpha=1, beta=1, out=buf1062)
    del arg605_1
    del arg606_1
    buf1063 = reinterpret_tensor(buf1062, (1, 128, 10240), (1310720, 10240, 1), 0); del buf1062  # reuse
    cpp_fused_gelu_230(c_void_p(buf1063.data_ptr()))
    buf1064 = reinterpret_tensor(buf1061, (128, 2560), (2560, 1), 0); del buf1061  # reuse
    # Source Nodes: [hidden_states_353], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg608_1, reinterpret_tensor(buf1063, (128, 10240), (10240, 1), 0), reinterpret_tensor(arg607_1, (10240, 2560), (1, 10240), 0), alpha=1, beta=1, out=buf1064)
    del arg607_1
    del arg608_1
    buf1065 = reinterpret_tensor(buf1064, (1, 128, 2560), (327680, 2560, 1), 0); del buf1064  # reuse
    buf1066 = buf1059; del buf1059  # reuse
    buf1067 = buf1058; del buf1058  # reuse
    buf1069 = reinterpret_tensor(buf1046, (1, 128, 2560), (327680, 2560, 1), 0); del buf1046  # reuse
    cpp_fused_add_native_layer_norm_231(c_void_p(buf1065.data_ptr()), c_void_p(buf1012.data_ptr()), c_void_p(buf1019.data_ptr()), c_void_p(buf1037.data_ptr()), c_void_p(buf1057.data_ptr()), c_void_p(arg609_1.data_ptr()), c_void_p(arg610_1.data_ptr()), c_void_p(buf1066.data_ptr()), c_void_p(buf1067.data_ptr()), c_void_p(buf1069.data_ptr()))
    del arg609_1
    del arg610_1
    buf1070 = buf1057; del buf1057  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_22_self_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg612_1, reinterpret_tensor(buf1069, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg611_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf1070)
    del arg611_1
    del arg612_1
    buf1071 = buf1037; del buf1037  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_22_self_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg614_1, reinterpret_tensor(buf1069, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg613_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf1071)
    del arg613_1
    del arg614_1
    buf1072 = reinterpret_tensor(buf1019, (1, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf1019  # reuse
    buf1073 = reinterpret_tensor(buf1012, (1, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf1012  # reuse
    cpp_fused_clone_232(c_void_p(buf1070.data_ptr()), c_void_p(buf1071.data_ptr()), c_void_p(buf1072.data_ptr()), c_void_p(buf1073.data_ptr()))
    buf1074 = buf1033; del buf1033  # reuse
    # Source Nodes: [attn_weights_136], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf1072, (32, 128, 80), (10240, 80, 1), 0), reinterpret_tensor(buf1073, (32, 80, 128), (10240, 1, 80), 0), out=buf1074)
    buf1075 = buf1031; del buf1031  # reuse
    buf1076 = buf1074; del buf1074  # reuse
    buf1077 = buf1029; del buf1029  # reuse
    cpp_fused__softmax_233(c_void_p(buf1076.data_ptr()), c_void_p(buf1075.data_ptr()), c_void_p(buf1077.data_ptr()))
    buf1078 = reinterpret_tensor(buf1073, (128, 2560), (2560, 1), 0); del buf1073  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_22_self_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg616_1, reinterpret_tensor(buf1069, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg615_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf1078)
    del arg615_1
    del arg616_1
    buf1079 = buf1076; del buf1076  # reuse
    buf1080 = reinterpret_tensor(buf1069, (1, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf1069  # reuse
    cpp_fused__softmax_clone_234(c_void_p(buf1079.data_ptr()), c_void_p(buf1077.data_ptr()), c_void_p(buf1078.data_ptr()), c_void_p(buf1080.data_ptr()))
    buf1081 = reinterpret_tensor(buf1078, (32, 128, 80), (10240, 80, 1), 0); del buf1078  # reuse
    # Source Nodes: [attn_output_230, attn_weights_139], Original ATen: [aten._softmax, aten.bmm]
    extern_kernels.bmm(buf1079, reinterpret_tensor(buf1080, (32, 128, 80), (10240, 80, 1), 0), out=buf1081)
    buf1082 = reinterpret_tensor(buf1080, (1, 128, 32, 80), (327680, 2560, 80, 1), 0); del buf1080  # reuse
    cpp_fused_clone_235(c_void_p(buf1081.data_ptr()), c_void_p(buf1082.data_ptr()))
    buf1083 = reinterpret_tensor(buf1081, (128, 2560), (2560, 1), 0); del buf1081  # reuse
    # Source Nodes: [hidden_states_358], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg618_1, reinterpret_tensor(buf1082, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg617_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf1083)
    del arg617_1
    del arg618_1
    buf1084 = buf1067; del buf1067  # reuse
    buf1085 = buf1066; del buf1066  # reuse
    buf1087 = reinterpret_tensor(buf1082, (1, 128, 2560), (327680, 2560, 1), 0); del buf1082  # reuse
    cpp_fused_add_native_layer_norm_236(c_void_p(buf1065.data_ptr()), c_void_p(buf1083.data_ptr()), c_void_p(arg619_1.data_ptr()), c_void_p(arg620_1.data_ptr()), c_void_p(buf1084.data_ptr()), c_void_p(buf1085.data_ptr()), c_void_p(buf1087.data_ptr()))
    del arg619_1
    del arg620_1
    buf1088 = reinterpret_tensor(buf1072, (128, 2560), (2560, 1), 0); del buf1072  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_22_encoder_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg622_1, reinterpret_tensor(buf1087, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg621_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf1088)
    del arg621_1
    del arg622_1
    buf1089 = reinterpret_tensor(buf1087, (128, 2560), (2560, 1), 0); del buf1087  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_22_encoder_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg624_1, reinterpret_tensor(buf81, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg623_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf1089)
    del arg623_1
    del arg624_1
    buf1090 = buf1071; del buf1071  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_22_encoder_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg626_1, reinterpret_tensor(buf81, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg625_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf1090)
    del arg625_1
    del arg626_1
    buf1091 = reinterpret_tensor(buf1070, (1, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf1070  # reuse
    buf1092 = buf1045; del buf1045  # reuse
    buf1093 = reinterpret_tensor(buf1044, (1, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf1044  # reuse
    cpp_fused_clone_237(c_void_p(buf1088.data_ptr()), c_void_p(buf1089.data_ptr()), c_void_p(buf1090.data_ptr()), c_void_p(buf1091.data_ptr()), c_void_p(buf1092.data_ptr()), c_void_p(buf1093.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf1094 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf1091, (1, 32, 128, 80), (0, 10240, 80, 1), 0), reinterpret_tensor(buf1092, (1, 32, 128, 80), (0, 10240, 80, 1), 0), reinterpret_tensor(buf1093, (1, 32, 128, 80), (0, 10240, 80, 1), 0), scale=1.0)
    buf1095 = buf1094[0]
    del buf1094
    buf1102 = reinterpret_tensor(buf1095, (1, 128, 32, 80), (327680, 2560, 80, 1), 0); del buf1095  # reuse
    cpp_fused_clone_238(c_void_p(buf1102.data_ptr()))
    buf1103 = reinterpret_tensor(buf1093, (128, 2560), (2560, 1), 0); del buf1093  # reuse
    # Source Nodes: [hidden_states_362], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg628_1, reinterpret_tensor(buf1102, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg627_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf1103)
    del arg627_1
    del arg628_1
    buf1104 = buf1085; del buf1085  # reuse
    buf1105 = buf1084; del buf1084  # reuse
    buf1107 = reinterpret_tensor(buf1102, (1, 128, 2560), (327680, 2560, 1), 0); del buf1102  # reuse
    cpp_fused_add_native_layer_norm_239(c_void_p(buf1065.data_ptr()), c_void_p(buf1083.data_ptr()), c_void_p(buf1103.data_ptr()), c_void_p(arg629_1.data_ptr()), c_void_p(arg630_1.data_ptr()), c_void_p(buf1104.data_ptr()), c_void_p(buf1105.data_ptr()), c_void_p(buf1107.data_ptr()))
    del arg629_1
    del arg630_1
    buf1108 = reinterpret_tensor(buf1063, (128, 10240), (10240, 1), 0); del buf1063  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_22_fc1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg632_1, reinterpret_tensor(buf1107, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg631_1, (2560, 10240), (1, 2560), 0), alpha=1, beta=1, out=buf1108)
    del arg631_1
    del arg632_1
    buf1109 = reinterpret_tensor(buf1108, (1, 128, 10240), (1310720, 10240, 1), 0); del buf1108  # reuse
    cpp_fused_gelu_240(c_void_p(buf1109.data_ptr()))
    buf1110 = reinterpret_tensor(buf1107, (128, 2560), (2560, 1), 0); del buf1107  # reuse
    # Source Nodes: [hidden_states_368], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg634_1, reinterpret_tensor(buf1109, (128, 10240), (10240, 1), 0), reinterpret_tensor(arg633_1, (10240, 2560), (1, 10240), 0), alpha=1, beta=1, out=buf1110)
    del arg633_1
    del arg634_1
    buf1111 = buf1105; del buf1105  # reuse
    buf1112 = buf1104; del buf1104  # reuse
    buf1114 = reinterpret_tensor(buf1092, (1, 128, 2560), (327680, 2560, 1), 0); del buf1092  # reuse
    cpp_fused_add_native_layer_norm_241(c_void_p(buf1065.data_ptr()), c_void_p(buf1083.data_ptr()), c_void_p(buf1103.data_ptr()), c_void_p(buf1110.data_ptr()), c_void_p(arg635_1.data_ptr()), c_void_p(arg636_1.data_ptr()), c_void_p(buf1111.data_ptr()), c_void_p(buf1112.data_ptr()), c_void_p(buf1114.data_ptr()))
    del arg635_1
    del arg636_1
    buf1115 = reinterpret_tensor(buf1091, (128, 2560), (2560, 1), 0); del buf1091  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_23_self_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg638_1, reinterpret_tensor(buf1114, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg637_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf1115)
    del arg637_1
    del arg638_1
    buf1116 = buf1090; del buf1090  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_23_self_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg640_1, reinterpret_tensor(buf1114, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg639_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf1116)
    del arg639_1
    del arg640_1
    buf1117 = reinterpret_tensor(buf1089, (1, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf1089  # reuse
    buf1118 = reinterpret_tensor(buf1088, (1, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf1088  # reuse
    cpp_fused_clone_242(c_void_p(buf1115.data_ptr()), c_void_p(buf1116.data_ptr()), c_void_p(buf1117.data_ptr()), c_void_p(buf1118.data_ptr()))
    del buf1115
    del buf1116
    buf1119 = buf1079; del buf1079  # reuse
    # Source Nodes: [attn_weights_142], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf1117, (32, 128, 80), (10240, 80, 1), 0), reinterpret_tensor(buf1118, (32, 80, 128), (10240, 1, 80), 0), out=buf1119)
    buf1120 = buf1077; del buf1077  # reuse
    buf1121 = buf1119; del buf1119  # reuse
    buf1122 = buf1075; del buf1075  # reuse
    cpp_fused__softmax_243(c_void_p(buf1121.data_ptr()), c_void_p(buf1120.data_ptr()), c_void_p(buf1122.data_ptr()))
    del buf1120
    buf1123 = reinterpret_tensor(buf1118, (128, 2560), (2560, 1), 0); del buf1118  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_23_self_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg642_1, reinterpret_tensor(buf1114, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg641_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf1123)
    del arg641_1
    del arg642_1
    buf1124 = buf1121; del buf1121  # reuse
    buf1125 = reinterpret_tensor(buf1114, (1, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf1114  # reuse
    cpp_fused__softmax_clone_244(c_void_p(buf1124.data_ptr()), c_void_p(buf1122.data_ptr()), c_void_p(buf1123.data_ptr()), c_void_p(buf1125.data_ptr()))
    del buf1122
    buf1126 = reinterpret_tensor(buf1123, (32, 128, 80), (10240, 80, 1), 0); del buf1123  # reuse
    # Source Nodes: [attn_output_240, attn_weights_145], Original ATen: [aten._softmax, aten.bmm]
    extern_kernels.bmm(buf1124, reinterpret_tensor(buf1125, (32, 128, 80), (10240, 80, 1), 0), out=buf1126)
    del buf1124
    buf1127 = reinterpret_tensor(buf1125, (1, 128, 32, 80), (327680, 2560, 80, 1), 0); del buf1125  # reuse
    cpp_fused_clone_245(c_void_p(buf1126.data_ptr()), c_void_p(buf1127.data_ptr()))
    buf1128 = reinterpret_tensor(buf1126, (128, 2560), (2560, 1), 0); del buf1126  # reuse
    # Source Nodes: [hidden_states_373], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg644_1, reinterpret_tensor(buf1127, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg643_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf1128)
    del arg643_1
    del arg644_1
    buf1129 = reinterpret_tensor(buf1128, (1, 128, 2560), (327680, 2560, 1), 0); del buf1128  # reuse
    buf1130 = buf1112; del buf1112  # reuse
    buf1131 = buf1111; del buf1111  # reuse
    buf1133 = reinterpret_tensor(buf1127, (1, 128, 2560), (327680, 2560, 1), 0); del buf1127  # reuse
    cpp_fused_add_native_layer_norm_246(c_void_p(buf1129.data_ptr()), c_void_p(buf1065.data_ptr()), c_void_p(buf1083.data_ptr()), c_void_p(buf1103.data_ptr()), c_void_p(buf1110.data_ptr()), c_void_p(arg645_1.data_ptr()), c_void_p(arg646_1.data_ptr()), c_void_p(buf1130.data_ptr()), c_void_p(buf1131.data_ptr()), c_void_p(buf1133.data_ptr()))
    del arg645_1
    del arg646_1
    buf1134 = buf1110; del buf1110  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_23_encoder_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg648_1, reinterpret_tensor(buf1133, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg647_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf1134)
    del arg647_1
    del arg648_1
    buf1135 = reinterpret_tensor(buf1133, (128, 2560), (2560, 1), 0); del buf1133  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_23_encoder_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg650_1, reinterpret_tensor(buf81, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg649_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf1135)
    del arg649_1
    del arg650_1
    buf1136 = buf1103; del buf1103  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_23_encoder_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg652_1, reinterpret_tensor(buf81, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg651_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf1136)
    del arg651_1
    del arg652_1
    buf1137 = reinterpret_tensor(buf1083, (1, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf1083  # reuse
    buf1138 = reinterpret_tensor(buf1065, (1, 32, 128, 80), (327680, 10240, 80, 1), 0); del buf1065  # reuse
    buf1139 = buf1117; del buf1117  # reuse
    cpp_fused_clone_247(c_void_p(buf1134.data_ptr()), c_void_p(buf1135.data_ptr()), c_void_p(buf1136.data_ptr()), c_void_p(buf1137.data_ptr()), c_void_p(buf1138.data_ptr()), c_void_p(buf1139.data_ptr()))
    del buf1134
    del buf1135
    del buf1136
    # Source Nodes: [], Original ATen: []
    buf1140 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf1137, (1, 32, 128, 80), (0, 10240, 80, 1), 0), reinterpret_tensor(buf1138, (1, 32, 128, 80), (0, 10240, 80, 1), 0), reinterpret_tensor(buf1139, (1, 32, 128, 80), (0, 10240, 80, 1), 0), scale=1.0)
    del buf1137
    buf1141 = buf1140[0]
    del buf1140
    buf1148 = reinterpret_tensor(buf1141, (1, 128, 32, 80), (327680, 2560, 80, 1), 0); del buf1141  # reuse
    cpp_fused_clone_248(c_void_p(buf1148.data_ptr()))
    buf1149 = reinterpret_tensor(buf1139, (128, 2560), (2560, 1), 0); del buf1139  # reuse
    # Source Nodes: [hidden_states_377], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg654_1, reinterpret_tensor(buf1148, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg653_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf1149)
    del arg653_1
    del arg654_1
    buf1150 = buf1131; del buf1131  # reuse
    buf1151 = buf1130; del buf1130  # reuse
    buf1153 = reinterpret_tensor(buf1148, (1, 128, 2560), (327680, 2560, 1), 0); del buf1148  # reuse
    cpp_fused_add_native_layer_norm_249(c_void_p(buf1129.data_ptr()), c_void_p(buf1149.data_ptr()), c_void_p(arg655_1.data_ptr()), c_void_p(arg656_1.data_ptr()), c_void_p(buf1150.data_ptr()), c_void_p(buf1151.data_ptr()), c_void_p(buf1153.data_ptr()))
    del arg655_1
    del arg656_1
    buf1154 = reinterpret_tensor(buf1109, (128, 10240), (10240, 1), 0); del buf1109  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_23_fc1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg658_1, reinterpret_tensor(buf1153, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg657_1, (2560, 10240), (1, 2560), 0), alpha=1, beta=1, out=buf1154)
    del arg657_1
    del arg658_1
    buf1155 = reinterpret_tensor(buf1154, (1, 128, 10240), (1310720, 10240, 1), 0); del buf1154  # reuse
    cpp_fused_gelu_250(c_void_p(buf1155.data_ptr()))
    buf1156 = reinterpret_tensor(buf1153, (128, 2560), (2560, 1), 0); del buf1153  # reuse
    # Source Nodes: [hidden_states_383], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg660_1, reinterpret_tensor(buf1155, (128, 10240), (10240, 1), 0), reinterpret_tensor(arg659_1, (10240, 2560), (1, 10240), 0), alpha=1, beta=1, out=buf1156)
    del arg659_1
    del arg660_1
    del buf1155
    buf1157 = buf1151; del buf1151  # reuse
    buf1158 = buf1150; del buf1150  # reuse
    buf1160 = reinterpret_tensor(buf1138, (1, 128, 2560), (327680, 2560, 1), 0); del buf1138  # reuse
    cpp_fused_add_native_layer_norm_251(c_void_p(buf1129.data_ptr()), c_void_p(buf1149.data_ptr()), c_void_p(buf1156.data_ptr()), c_void_p(arg661_1.data_ptr()), c_void_p(arg662_1.data_ptr()), c_void_p(buf1157.data_ptr()), c_void_p(buf1158.data_ptr()), c_void_p(buf1160.data_ptr()))
    del arg661_1
    del arg662_1
    del buf1129
    del buf1149
    del buf1156
    buf1161 = empty((128, 8008), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___lm_head], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf1160, (128, 2560), (2560, 1), 0), reinterpret_tensor(arg663_1, (2560, 8008), (1, 2560), 0), out=buf1161)
    del arg663_1
    del buf1160
    buf1162 = reinterpret_tensor(buf1161, (1, 128, 8008), (1025024, 8008, 1), 0); del buf1161  # reuse
    buf1163 = reinterpret_tensor(buf1158, (128, 1), (1, 128), 0); del buf1158  # reuse
    buf1164 = reinterpret_tensor(buf1157, (128, 1), (1, 128), 0); del buf1157  # reuse
    buf1165 = empty((), device='cpu', dtype=torch.float32)
    buf1166 = empty((), device='cpu', dtype=torch.int64)
    buf1167 = buf1165; del buf1165  # reuse
    cpp_fused__log_softmax_add_nll_loss_forward_252(c_void_p(buf1162.data_ptr()), c_void_p(buf1167.data_ptr()), c_void_p(arg664_1.data_ptr()), c_void_p(arg665_1.data_ptr()), c_void_p(buf1163.data_ptr()), c_void_p(buf1164.data_ptr()), c_void_p(buf1166.data_ptr()))
    del arg664_1
    del arg665_1
    return (buf1167, buf1162, buf81, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((128, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg1_1 = rand_strided((128, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg2_1 = rand_strided((8008, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg3_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg4_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg5_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg6_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg7_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg8_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg9_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg10_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg11_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg12_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg13_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg14_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg15_1 = rand_strided((10240, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg16_1 = rand_strided((10240, ), (1, ), device='cpu', dtype=torch.float32)
    arg17_1 = rand_strided((2560, 10240), (10240, 1), device='cpu', dtype=torch.float32)
    arg18_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg19_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg20_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg21_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg22_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg23_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg24_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg25_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg26_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg27_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg28_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg29_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg30_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg31_1 = rand_strided((10240, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg32_1 = rand_strided((10240, ), (1, ), device='cpu', dtype=torch.float32)
    arg33_1 = rand_strided((2560, 10240), (10240, 1), device='cpu', dtype=torch.float32)
    arg34_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg35_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg36_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg37_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg38_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg39_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg40_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg41_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg42_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg43_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg44_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg45_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg46_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg47_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg48_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg49_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg50_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg51_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg52_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg53_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg54_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg55_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg56_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg57_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg58_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg59_1 = rand_strided((10240, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg60_1 = rand_strided((10240, ), (1, ), device='cpu', dtype=torch.float32)
    arg61_1 = rand_strided((2560, 10240), (10240, 1), device='cpu', dtype=torch.float32)
    arg62_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg63_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg64_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg65_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg66_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg67_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg68_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg69_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg70_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg71_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg72_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg73_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg74_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg75_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg76_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg77_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg78_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg79_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg80_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg81_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg82_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg83_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg84_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg85_1 = rand_strided((10240, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg86_1 = rand_strided((10240, ), (1, ), device='cpu', dtype=torch.float32)
    arg87_1 = rand_strided((2560, 10240), (10240, 1), device='cpu', dtype=torch.float32)
    arg88_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg89_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg90_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg91_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg92_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg93_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg94_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg95_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg96_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg97_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg98_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg99_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg100_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg101_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg102_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg103_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg104_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg105_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg106_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg107_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg108_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg109_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg110_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg111_1 = rand_strided((10240, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg112_1 = rand_strided((10240, ), (1, ), device='cpu', dtype=torch.float32)
    arg113_1 = rand_strided((2560, 10240), (10240, 1), device='cpu', dtype=torch.float32)
    arg114_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg115_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg116_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg117_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg118_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg119_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg120_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg121_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg122_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg123_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg124_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg125_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg126_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg127_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg128_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg129_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg130_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg131_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg132_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg133_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg134_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg135_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg136_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg137_1 = rand_strided((10240, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg138_1 = rand_strided((10240, ), (1, ), device='cpu', dtype=torch.float32)
    arg139_1 = rand_strided((2560, 10240), (10240, 1), device='cpu', dtype=torch.float32)
    arg140_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg141_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg142_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg143_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg144_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg145_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg146_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg147_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg148_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg149_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg150_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg151_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg152_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg153_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg154_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg155_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg156_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg157_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg158_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg159_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg160_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg161_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg162_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg163_1 = rand_strided((10240, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg164_1 = rand_strided((10240, ), (1, ), device='cpu', dtype=torch.float32)
    arg165_1 = rand_strided((2560, 10240), (10240, 1), device='cpu', dtype=torch.float32)
    arg166_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg167_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg168_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg169_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg170_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg171_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg172_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg173_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg174_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg175_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg176_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg177_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg178_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg179_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg180_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg181_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg182_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg183_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg184_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg185_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg186_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg187_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg188_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg189_1 = rand_strided((10240, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg190_1 = rand_strided((10240, ), (1, ), device='cpu', dtype=torch.float32)
    arg191_1 = rand_strided((2560, 10240), (10240, 1), device='cpu', dtype=torch.float32)
    arg192_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg193_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg194_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg195_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg196_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg197_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg198_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg199_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg200_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg201_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg202_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg203_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg204_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg205_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg206_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg207_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg208_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg209_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg210_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg211_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg212_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg213_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg214_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg215_1 = rand_strided((10240, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg216_1 = rand_strided((10240, ), (1, ), device='cpu', dtype=torch.float32)
    arg217_1 = rand_strided((2560, 10240), (10240, 1), device='cpu', dtype=torch.float32)
    arg218_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg219_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg220_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg221_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg222_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg223_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg224_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg225_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg226_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg227_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg228_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg229_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg230_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg231_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg232_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg233_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg234_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg235_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg236_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg237_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg238_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg239_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg240_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg241_1 = rand_strided((10240, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg242_1 = rand_strided((10240, ), (1, ), device='cpu', dtype=torch.float32)
    arg243_1 = rand_strided((2560, 10240), (10240, 1), device='cpu', dtype=torch.float32)
    arg244_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg245_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg246_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg247_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg248_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg249_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg250_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg251_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg252_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg253_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg254_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg255_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg256_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg257_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg258_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg259_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg260_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg261_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg262_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg263_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg264_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg265_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg266_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg267_1 = rand_strided((10240, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg268_1 = rand_strided((10240, ), (1, ), device='cpu', dtype=torch.float32)
    arg269_1 = rand_strided((2560, 10240), (10240, 1), device='cpu', dtype=torch.float32)
    arg270_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg271_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg272_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg273_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg274_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg275_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg276_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg277_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg278_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg279_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg280_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg281_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg282_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg283_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg284_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg285_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg286_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg287_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg288_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg289_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg290_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg291_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg292_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg293_1 = rand_strided((10240, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg294_1 = rand_strided((10240, ), (1, ), device='cpu', dtype=torch.float32)
    arg295_1 = rand_strided((2560, 10240), (10240, 1), device='cpu', dtype=torch.float32)
    arg296_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg297_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg298_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg299_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg300_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg301_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg302_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg303_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg304_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg305_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg306_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg307_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg308_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg309_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg310_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg311_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg312_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg313_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg314_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg315_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg316_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg317_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg318_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg319_1 = rand_strided((10240, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg320_1 = rand_strided((10240, ), (1, ), device='cpu', dtype=torch.float32)
    arg321_1 = rand_strided((2560, 10240), (10240, 1), device='cpu', dtype=torch.float32)
    arg322_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg323_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg324_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg325_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg326_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg327_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg328_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg329_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg330_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg331_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg332_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg333_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg334_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg335_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg336_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg337_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg338_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg339_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg340_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg341_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg342_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg343_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg344_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg345_1 = rand_strided((10240, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg346_1 = rand_strided((10240, ), (1, ), device='cpu', dtype=torch.float32)
    arg347_1 = rand_strided((2560, 10240), (10240, 1), device='cpu', dtype=torch.float32)
    arg348_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg349_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg350_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg351_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg352_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg353_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg354_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg355_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg356_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg357_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg358_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg359_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg360_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg361_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg362_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg363_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg364_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg365_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg366_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg367_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg368_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg369_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg370_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg371_1 = rand_strided((10240, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg372_1 = rand_strided((10240, ), (1, ), device='cpu', dtype=torch.float32)
    arg373_1 = rand_strided((2560, 10240), (10240, 1), device='cpu', dtype=torch.float32)
    arg374_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg375_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg376_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg377_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg378_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg379_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg380_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg381_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg382_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg383_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg384_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg385_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg386_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg387_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg388_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg389_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg390_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg391_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg392_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg393_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg394_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg395_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg396_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg397_1 = rand_strided((10240, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg398_1 = rand_strided((10240, ), (1, ), device='cpu', dtype=torch.float32)
    arg399_1 = rand_strided((2560, 10240), (10240, 1), device='cpu', dtype=torch.float32)
    arg400_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg401_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg402_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg403_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg404_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg405_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg406_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg407_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg408_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg409_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg410_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg411_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg412_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg413_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg414_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg415_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg416_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg417_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg418_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg419_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg420_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg421_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg422_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg423_1 = rand_strided((10240, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg424_1 = rand_strided((10240, ), (1, ), device='cpu', dtype=torch.float32)
    arg425_1 = rand_strided((2560, 10240), (10240, 1), device='cpu', dtype=torch.float32)
    arg426_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg427_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg428_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg429_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg430_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg431_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg432_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg433_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg434_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg435_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg436_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg437_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg438_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg439_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg440_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg441_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg442_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg443_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg444_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg445_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg446_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg447_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg448_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg449_1 = rand_strided((10240, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg450_1 = rand_strided((10240, ), (1, ), device='cpu', dtype=torch.float32)
    arg451_1 = rand_strided((2560, 10240), (10240, 1), device='cpu', dtype=torch.float32)
    arg452_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg453_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg454_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg455_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg456_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg457_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg458_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg459_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg460_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg461_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg462_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg463_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg464_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg465_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg466_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg467_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg468_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg469_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg470_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg471_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg472_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg473_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg474_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg475_1 = rand_strided((10240, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg476_1 = rand_strided((10240, ), (1, ), device='cpu', dtype=torch.float32)
    arg477_1 = rand_strided((2560, 10240), (10240, 1), device='cpu', dtype=torch.float32)
    arg478_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg479_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg480_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg481_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg482_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg483_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg484_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg485_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg486_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg487_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg488_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg489_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg490_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg491_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg492_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg493_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg494_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg495_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg496_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg497_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg498_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg499_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg500_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg501_1 = rand_strided((10240, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg502_1 = rand_strided((10240, ), (1, ), device='cpu', dtype=torch.float32)
    arg503_1 = rand_strided((2560, 10240), (10240, 1), device='cpu', dtype=torch.float32)
    arg504_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg505_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg506_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg507_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg508_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg509_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg510_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg511_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg512_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg513_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg514_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg515_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg516_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg517_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg518_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg519_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg520_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg521_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg522_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg523_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg524_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg525_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg526_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg527_1 = rand_strided((10240, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg528_1 = rand_strided((10240, ), (1, ), device='cpu', dtype=torch.float32)
    arg529_1 = rand_strided((2560, 10240), (10240, 1), device='cpu', dtype=torch.float32)
    arg530_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg531_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg532_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg533_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg534_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg535_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg536_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg537_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg538_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg539_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg540_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg541_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg542_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg543_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg544_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg545_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg546_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg547_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg548_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg549_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg550_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg551_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg552_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg553_1 = rand_strided((10240, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg554_1 = rand_strided((10240, ), (1, ), device='cpu', dtype=torch.float32)
    arg555_1 = rand_strided((2560, 10240), (10240, 1), device='cpu', dtype=torch.float32)
    arg556_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg557_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg558_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg559_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg560_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg561_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg562_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg563_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg564_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg565_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg566_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg567_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg568_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg569_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg570_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg571_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg572_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg573_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg574_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg575_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg576_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg577_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg578_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg579_1 = rand_strided((10240, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg580_1 = rand_strided((10240, ), (1, ), device='cpu', dtype=torch.float32)
    arg581_1 = rand_strided((2560, 10240), (10240, 1), device='cpu', dtype=torch.float32)
    arg582_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg583_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg584_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg585_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg586_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg587_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg588_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg589_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg590_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg591_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg592_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg593_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg594_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg595_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg596_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg597_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg598_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg599_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg600_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg601_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg602_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg603_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg604_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg605_1 = rand_strided((10240, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg606_1 = rand_strided((10240, ), (1, ), device='cpu', dtype=torch.float32)
    arg607_1 = rand_strided((2560, 10240), (10240, 1), device='cpu', dtype=torch.float32)
    arg608_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg609_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg610_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg611_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg612_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg613_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg614_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg615_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg616_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg617_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg618_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg619_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg620_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg621_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg622_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg623_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg624_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg625_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg626_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg627_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg628_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg629_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg630_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg631_1 = rand_strided((10240, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg632_1 = rand_strided((10240, ), (1, ), device='cpu', dtype=torch.float32)
    arg633_1 = rand_strided((2560, 10240), (10240, 1), device='cpu', dtype=torch.float32)
    arg634_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg635_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg636_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg637_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg638_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg639_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg640_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg641_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg642_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg643_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg644_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg645_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg646_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg647_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg648_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg649_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg650_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg651_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg652_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg653_1 = rand_strided((2560, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg654_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg655_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg656_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg657_1 = rand_strided((10240, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg658_1 = rand_strided((10240, ), (1, ), device='cpu', dtype=torch.float32)
    arg659_1 = rand_strided((2560, 10240), (10240, 1), device='cpu', dtype=torch.float32)
    arg660_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg661_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg662_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg663_1 = rand_strided((8008, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg664_1 = rand_strided((1, 8008), (8008, 1), device='cpu', dtype=torch.float32)
    arg665_1 = rand_strided((1, 128), (128, 1), device='cpu', dtype=torch.int64)
    arg666_1 = rand_strided((1, 128), (128, 1), device='cpu', dtype=torch.int64)
    arg667_1 = rand_strided((1, 128), (128, 1), device='cpu', dtype=torch.int64)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1, arg473_1, arg474_1, arg475_1, arg476_1, arg477_1, arg478_1, arg479_1, arg480_1, arg481_1, arg482_1, arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, arg503_1, arg504_1, arg505_1, arg506_1, arg507_1, arg508_1, arg509_1, arg510_1, arg511_1, arg512_1, arg513_1, arg514_1, arg515_1, arg516_1, arg517_1, arg518_1, arg519_1, arg520_1, arg521_1, arg522_1, arg523_1, arg524_1, arg525_1, arg526_1, arg527_1, arg528_1, arg529_1, arg530_1, arg531_1, arg532_1, arg533_1, arg534_1, arg535_1, arg536_1, arg537_1, arg538_1, arg539_1, arg540_1, arg541_1, arg542_1, arg543_1, arg544_1, arg545_1, arg546_1, arg547_1, arg548_1, arg549_1, arg550_1, arg551_1, arg552_1, arg553_1, arg554_1, arg555_1, arg556_1, arg557_1, arg558_1, arg559_1, arg560_1, arg561_1, arg562_1, arg563_1, arg564_1, arg565_1, arg566_1, arg567_1, arg568_1, arg569_1, arg570_1, arg571_1, arg572_1, arg573_1, arg574_1, arg575_1, arg576_1, arg577_1, arg578_1, arg579_1, arg580_1, arg581_1, arg582_1, arg583_1, arg584_1, arg585_1, arg586_1, arg587_1, arg588_1, arg589_1, arg590_1, arg591_1, arg592_1, arg593_1, arg594_1, arg595_1, arg596_1, arg597_1, arg598_1, arg599_1, arg600_1, arg601_1, arg602_1, arg603_1, arg604_1, arg605_1, arg606_1, arg607_1, arg608_1, arg609_1, arg610_1, arg611_1, arg612_1, arg613_1, arg614_1, arg615_1, arg616_1, arg617_1, arg618_1, arg619_1, arg620_1, arg621_1, arg622_1, arg623_1, arg624_1, arg625_1, arg626_1, arg627_1, arg628_1, arg629_1, arg630_1, arg631_1, arg632_1, arg633_1, arg634_1, arg635_1, arg636_1, arg637_1, arg638_1, arg639_1, arg640_1, arg641_1, arg642_1, arg643_1, arg644_1, arg645_1, arg646_1, arg647_1, arg648_1, arg649_1, arg650_1, arg651_1, arg652_1, arg653_1, arg654_1, arg655_1, arg656_1, arg657_1, arg658_1, arg659_1, arg660_1, arg661_1, arg662_1, arg663_1, arg664_1, arg665_1, arg666_1, arg667_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('BlenderbotForConditionalGeneration', benchmark_compiled_module)
