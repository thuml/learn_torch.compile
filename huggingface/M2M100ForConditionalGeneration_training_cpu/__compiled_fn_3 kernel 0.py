
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


cpp_fused__to_copy_cumsum_ne_0 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const long* in_ptr0,
                       int* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = in_ptr0[static_cast<long>(x0)];
            auto tmp1 = static_cast<long>(1);
            auto tmp2 = tmp0 != tmp1;
            auto tmp3 = c10::convert<int>(tmp2);
            out_ptr0[static_cast<long>(x0)] = tmp3;
        }
    }
}
''')


cpp_fused_add_embedding_mul_native_layer_norm_view_1 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const long* in_ptr0,
                       const float* in_ptr1,
                       const long* in_ptr2,
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
                    #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0)];
                        auto tmp7 = in_ptr2[static_cast<long>(x0)];
                        auto tmp1 = decltype(tmp0)(tmp0 + 128112);
                        auto tmp2 = tmp0 < 0;
                        auto tmp3 = tmp2 ? tmp1 : tmp0;
                        TORCH_CHECK((0 <= tmp3) & (tmp3 < 128112L), "index out of bounds: 0 <= tmp3 < 128112L")
                        auto tmp4 = in_ptr1[static_cast<long>(x1 + (1024L*tmp3))];
                        auto tmp5 = static_cast<float>(32.0);
                        auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                        auto tmp8 = c10::convert<int>(tmp7);
                        auto tmp9 = static_cast<int>(0);
                        auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                        auto tmp11 = static_cast<long>(1);
                        auto tmp12 = tmp0 != tmp11;
                        auto tmp13 = c10::convert<int>(tmp12);
                        auto tmp14 = decltype(tmp10)(tmp10 * tmp13);
                        auto tmp15 = c10::convert<long>(tmp14);
                        auto tmp16 = decltype(tmp15)(tmp15 + tmp11);
                        auto tmp17 = decltype(tmp16)(tmp16 + 1026);
                        auto tmp18 = tmp16 < 0;
                        auto tmp19 = tmp18 ? tmp17 : tmp16;
                        TORCH_CHECK((0 <= tmp19) & (tmp19 < 1026L), "index out of bounds: 0 <= tmp19 < 1026L")
                        auto tmp20 = in_ptr3[static_cast<long>(x1 + (1024L*tmp19))];
                        auto tmp21 = decltype(tmp6)(tmp6 + tmp20);
                        tmp_acc0 = welford_combine(tmp_acc0, tmp21);
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0.mean;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc0.m2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x0)];
                    auto tmp7 = in_ptr2[static_cast<long>(x0)];
                    auto tmp22 = out_ptr0[static_cast<long>(x0)];
                    auto tmp24 = out_ptr1[static_cast<long>(x0)];
                    auto tmp31 = in_ptr4[static_cast<long>(x1)];
                    auto tmp33 = in_ptr5[static_cast<long>(x1)];
                    auto tmp1 = decltype(tmp0)(tmp0 + 128112);
                    auto tmp2 = tmp0 < 0;
                    auto tmp3 = tmp2 ? tmp1 : tmp0;
                    TORCH_CHECK((0 <= tmp3) & (tmp3 < 128112L), "index out of bounds: 0 <= tmp3 < 128112L")
                    auto tmp4 = in_ptr1[static_cast<long>(x1 + (1024L*tmp3))];
                    auto tmp5 = static_cast<float>(32.0);
                    auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                    auto tmp8 = c10::convert<int>(tmp7);
                    auto tmp9 = static_cast<int>(0);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = static_cast<long>(1);
                    auto tmp12 = tmp0 != tmp11;
                    auto tmp13 = c10::convert<int>(tmp12);
                    auto tmp14 = decltype(tmp10)(tmp10 * tmp13);
                    auto tmp15 = c10::convert<long>(tmp14);
                    auto tmp16 = decltype(tmp15)(tmp15 + tmp11);
                    auto tmp17 = decltype(tmp16)(tmp16 + 1026);
                    auto tmp18 = tmp16 < 0;
                    auto tmp19 = tmp18 ? tmp17 : tmp16;
                    TORCH_CHECK((0 <= tmp19) & (tmp19 < 1026L), "index out of bounds: 0 <= tmp19 < 1026L")
                    auto tmp20 = in_ptr3[static_cast<long>(x1 + (1024L*tmp19))];
                    auto tmp21 = decltype(tmp6)(tmp6 + tmp20);
                    auto tmp23 = decltype(tmp21)(tmp21 - tmp22);
                    auto tmp25 = static_cast<float>(1024.0);
                    auto tmp26 = tmp24 / tmp25;
                    auto tmp27 = static_cast<float>(1e-05);
                    auto tmp28 = decltype(tmp26)(tmp26 + tmp27);
                    auto tmp29 = 1 / std::sqrt(tmp28);
                    auto tmp30 = decltype(tmp23)(tmp23 * tmp29);
                    auto tmp32 = decltype(tmp30)(tmp30 * tmp31);
                    auto tmp34 = decltype(tmp32)(tmp32 + tmp33);
                    out_ptr2[static_cast<long>(x1 + (1024L*x0))] = tmp30;
                    out_ptr3[static_cast<long>(x1 + (1024L*x0))] = tmp34;
                }
            }
        }
    }
}
''')


cpp_fused_clone_2 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_3 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto out_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp1 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp4 = tmp3.exp();
                        tmp4.store(out_ptr1 + static_cast<long>(x1 + (128L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = out_ptr2[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr3 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_4 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_embedding_mul_native_layer_norm_view_5 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const long* in_ptr0,
                       const float* in_ptr1,
                       const long* in_ptr2,
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
                    #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0)];
                        auto tmp7 = in_ptr2[static_cast<long>(x0)];
                        auto tmp22 = in_out_ptr0[static_cast<long>(x1 + (1024L*x0))];
                        auto tmp1 = decltype(tmp0)(tmp0 + 128112);
                        auto tmp2 = tmp0 < 0;
                        auto tmp3 = tmp2 ? tmp1 : tmp0;
                        TORCH_CHECK((0 <= tmp3) & (tmp3 < 128112L), "index out of bounds: 0 <= tmp3 < 128112L")
                        auto tmp4 = in_ptr1[static_cast<long>(x1 + (1024L*tmp3))];
                        auto tmp5 = static_cast<float>(32.0);
                        auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                        auto tmp8 = c10::convert<int>(tmp7);
                        auto tmp9 = static_cast<int>(0);
                        auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                        auto tmp11 = static_cast<long>(1);
                        auto tmp12 = tmp0 != tmp11;
                        auto tmp13 = c10::convert<int>(tmp12);
                        auto tmp14 = decltype(tmp10)(tmp10 * tmp13);
                        auto tmp15 = c10::convert<long>(tmp14);
                        auto tmp16 = decltype(tmp15)(tmp15 + tmp11);
                        auto tmp17 = decltype(tmp16)(tmp16 + 1026);
                        auto tmp18 = tmp16 < 0;
                        auto tmp19 = tmp18 ? tmp17 : tmp16;
                        TORCH_CHECK((0 <= tmp19) & (tmp19 < 1026L), "index out of bounds: 0 <= tmp19 < 1026L")
                        auto tmp20 = in_ptr3[static_cast<long>(x1 + (1024L*tmp19))];
                        auto tmp21 = decltype(tmp6)(tmp6 + tmp20);
                        auto tmp23 = decltype(tmp21)(tmp21 + tmp22);
                        in_out_ptr0[static_cast<long>(x1 + (1024L*x0))] = tmp23;
                        tmp_acc0 = welford_combine(tmp_acc0, tmp23);
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0.mean;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc0.m2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_relu_view_6 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                tmp1.store(out_ptr0 + static_cast<long>(x0));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp13.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_8 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_9 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto out_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp1 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp4 = tmp3.exp();
                        tmp4.store(out_ptr1 + static_cast<long>(x1 + (128L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = out_ptr2[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr3 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_10 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                    auto tmp11 = static_cast<float>(1e-05);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    tmp19.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_relu_view_12 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                tmp1.store(out_ptr0 + static_cast<long>(x0));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                    auto tmp13 = static_cast<float>(1e-05);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_14 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_15 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto out_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp1 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp4 = tmp3.exp();
                        tmp4.store(out_ptr1 + static_cast<long>(x1 + (128L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = out_ptr2[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr3 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_16 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_17 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_relu_view_18 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                tmp1.store(out_ptr0 + static_cast<long>(x0));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp13.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
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
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_21 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto out_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp1 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp4 = tmp3.exp();
                        tmp4.store(out_ptr1 + static_cast<long>(x1 + (128L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = out_ptr2[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr3 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    }
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                    auto tmp11 = static_cast<float>(1e-05);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    tmp19.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_relu_view_24 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                tmp1.store(out_ptr0 + static_cast<long>(x0));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                    auto tmp13 = static_cast<float>(1e-05);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_26 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_27 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto out_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp1 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp4 = tmp3.exp();
                        tmp4.store(out_ptr1 + static_cast<long>(x1 + (128L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = out_ptr2[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr3 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    }
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_29 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_relu_view_30 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                tmp1.store(out_ptr0 + static_cast<long>(x0));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp13.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_33 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto out_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp1 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp4 = tmp3.exp();
                        tmp4.store(out_ptr1 + static_cast<long>(x1 + (128L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = out_ptr2[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr3 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_34 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                    auto tmp11 = static_cast<float>(1e-05);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    tmp19.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_relu_view_36 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_37 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                    auto tmp13 = static_cast<float>(1e-05);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_38 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_39 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto out_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp1 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp4 = tmp3.exp();
                        tmp4.store(out_ptr1 + static_cast<long>(x1 + (128L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = out_ptr2[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr3 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_40 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_41 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_relu_view_42 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                tmp1.store(out_ptr0 + static_cast<long>(x0));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp13.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_44 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_45 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto out_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp1 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp4 = tmp3.exp();
                        tmp4.store(out_ptr1 + static_cast<long>(x1 + (128L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = out_ptr2[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr3 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_46 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_47 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                    auto tmp11 = static_cast<float>(1e-05);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    tmp19.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_relu_view_48 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_49 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                    auto tmp13 = static_cast<float>(1e-05);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
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
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_51 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto out_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp1 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp4 = tmp3.exp();
                        tmp4.store(out_ptr1 + static_cast<long>(x1 + (128L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = out_ptr2[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr3 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    }
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_53 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_relu_view_54 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                tmp1.store(out_ptr0 + static_cast<long>(x0));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp13.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_56 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_57 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto out_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp1 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp4 = tmp3.exp();
                        tmp4.store(out_ptr1 + static_cast<long>(x1 + (128L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = out_ptr2[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr3 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    }
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_59 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                    auto tmp11 = static_cast<float>(1e-05);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    tmp19.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_relu_view_60 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_61 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                    auto tmp13 = static_cast<float>(1e-05);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_63 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto out_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp1 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp4 = tmp3.exp();
                        tmp4.store(out_ptr1 + static_cast<long>(x1 + (128L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = out_ptr2[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr3 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_64 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_relu_view_66 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_67 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp13.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_68 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_69 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto out_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp1 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp4 = tmp3.exp();
                        tmp4.store(out_ptr1 + static_cast<long>(x1 + (128L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = out_ptr2[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr3 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_70 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_71 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                    auto tmp11 = static_cast<float>(1e-05);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    tmp19.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_relu_view_72 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__to_copy_add_cumsum_native_layer_norm_ne_73 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const long* in_ptr6,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       int* out_ptr4)
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                    auto tmp13 = static_cast<float>(1e-05);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr6[static_cast<long>(x0)];
                    auto tmp1 = static_cast<long>(1);
                    auto tmp2 = tmp0 != tmp1;
                    auto tmp3 = c10::convert<int>(tmp2);
                    out_ptr4[static_cast<long>(x0)] = tmp3;
                }
            }
        }
    }
}
''')


cpp_fused_add_embedding_mul_native_layer_norm_view_74 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const long* in_ptr0,
                       const float* in_ptr1,
                       const long* in_ptr2,
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
                    #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0)];
                        auto tmp7 = in_ptr2[static_cast<long>(x0)];
                        auto tmp1 = decltype(tmp0)(tmp0 + 128112);
                        auto tmp2 = tmp0 < 0;
                        auto tmp3 = tmp2 ? tmp1 : tmp0;
                        TORCH_CHECK((0 <= tmp3) & (tmp3 < 128112L), "index out of bounds: 0 <= tmp3 < 128112L")
                        auto tmp4 = in_ptr1[static_cast<long>(x1 + (1024L*tmp3))];
                        auto tmp5 = static_cast<float>(32.0);
                        auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                        auto tmp8 = c10::convert<int>(tmp7);
                        auto tmp9 = static_cast<int>(0);
                        auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                        auto tmp11 = static_cast<long>(1);
                        auto tmp12 = tmp0 != tmp11;
                        auto tmp13 = c10::convert<int>(tmp12);
                        auto tmp14 = decltype(tmp10)(tmp10 * tmp13);
                        auto tmp15 = c10::convert<long>(tmp14);
                        auto tmp16 = decltype(tmp15)(tmp15 + tmp11);
                        auto tmp17 = decltype(tmp16)(tmp16 + 1026);
                        auto tmp18 = tmp16 < 0;
                        auto tmp19 = tmp18 ? tmp17 : tmp16;
                        TORCH_CHECK((0 <= tmp19) & (tmp19 < 1026L), "index out of bounds: 0 <= tmp19 < 1026L")
                        auto tmp20 = in_ptr3[static_cast<long>(x1 + (1024L*tmp19))];
                        auto tmp21 = decltype(tmp6)(tmp6 + tmp20);
                        tmp_acc0 = welford_combine(tmp_acc0, tmp21);
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0.mean;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc0.m2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x0)];
                    auto tmp7 = in_ptr2[static_cast<long>(x0)];
                    auto tmp22 = out_ptr0[static_cast<long>(x0)];
                    auto tmp24 = out_ptr1[static_cast<long>(x0)];
                    auto tmp31 = in_ptr4[static_cast<long>(x1)];
                    auto tmp33 = in_ptr5[static_cast<long>(x1)];
                    auto tmp1 = decltype(tmp0)(tmp0 + 128112);
                    auto tmp2 = tmp0 < 0;
                    auto tmp3 = tmp2 ? tmp1 : tmp0;
                    TORCH_CHECK((0 <= tmp3) & (tmp3 < 128112L), "index out of bounds: 0 <= tmp3 < 128112L")
                    auto tmp4 = in_ptr1[static_cast<long>(x1 + (1024L*tmp3))];
                    auto tmp5 = static_cast<float>(32.0);
                    auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                    auto tmp8 = c10::convert<int>(tmp7);
                    auto tmp9 = static_cast<int>(0);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = static_cast<long>(1);
                    auto tmp12 = tmp0 != tmp11;
                    auto tmp13 = c10::convert<int>(tmp12);
                    auto tmp14 = decltype(tmp10)(tmp10 * tmp13);
                    auto tmp15 = c10::convert<long>(tmp14);
                    auto tmp16 = decltype(tmp15)(tmp15 + tmp11);
                    auto tmp17 = decltype(tmp16)(tmp16 + 1026);
                    auto tmp18 = tmp16 < 0;
                    auto tmp19 = tmp18 ? tmp17 : tmp16;
                    TORCH_CHECK((0 <= tmp19) & (tmp19 < 1026L), "index out of bounds: 0 <= tmp19 < 1026L")
                    auto tmp20 = in_ptr3[static_cast<long>(x1 + (1024L*tmp19))];
                    auto tmp21 = decltype(tmp6)(tmp6 + tmp20);
                    auto tmp23 = decltype(tmp21)(tmp21 - tmp22);
                    auto tmp25 = static_cast<float>(1024.0);
                    auto tmp26 = tmp24 / tmp25;
                    auto tmp27 = static_cast<float>(1e-05);
                    auto tmp28 = decltype(tmp26)(tmp26 + tmp27);
                    auto tmp29 = 1 / std::sqrt(tmp28);
                    auto tmp30 = decltype(tmp23)(tmp23 * tmp29);
                    auto tmp32 = decltype(tmp30)(tmp30 * tmp31);
                    auto tmp34 = decltype(tmp32)(tmp32 + tmp33);
                    out_ptr2[static_cast<long>(x1 + (1024L*x0))] = tmp30;
                    out_ptr3[static_cast<long>(x1 + (1024L*x0))] = tmp34;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_76 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_77 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_embedding_mul_native_layer_norm_view_79 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const long* in_ptr0,
                       const float* in_ptr1,
                       const long* in_ptr2,
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
                    #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0)];
                        auto tmp7 = in_ptr2[static_cast<long>(x0)];
                        auto tmp22 = in_out_ptr0[static_cast<long>(x1 + (1024L*x0))];
                        auto tmp1 = decltype(tmp0)(tmp0 + 128112);
                        auto tmp2 = tmp0 < 0;
                        auto tmp3 = tmp2 ? tmp1 : tmp0;
                        TORCH_CHECK((0 <= tmp3) & (tmp3 < 128112L), "index out of bounds: 0 <= tmp3 < 128112L")
                        auto tmp4 = in_ptr1[static_cast<long>(x1 + (1024L*tmp3))];
                        auto tmp5 = static_cast<float>(32.0);
                        auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                        auto tmp8 = c10::convert<int>(tmp7);
                        auto tmp9 = static_cast<int>(0);
                        auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                        auto tmp11 = static_cast<long>(1);
                        auto tmp12 = tmp0 != tmp11;
                        auto tmp13 = c10::convert<int>(tmp12);
                        auto tmp14 = decltype(tmp10)(tmp10 * tmp13);
                        auto tmp15 = c10::convert<long>(tmp14);
                        auto tmp16 = decltype(tmp15)(tmp15 + tmp11);
                        auto tmp17 = decltype(tmp16)(tmp16 + 1026);
                        auto tmp18 = tmp16 < 0;
                        auto tmp19 = tmp18 ? tmp17 : tmp16;
                        TORCH_CHECK((0 <= tmp19) & (tmp19 < 1026L), "index out of bounds: 0 <= tmp19 < 1026L")
                        auto tmp20 = in_ptr3[static_cast<long>(x1 + (1024L*tmp19))];
                        auto tmp21 = decltype(tmp6)(tmp6 + tmp20);
                        auto tmp23 = decltype(tmp21)(tmp21 + tmp22);
                        in_out_ptr0[static_cast<long>(x1 + (1024L*x0))] = tmp23;
                        tmp_acc0 = welford_combine(tmp_acc0, tmp23);
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0.mean;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc0.m2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_80 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_81 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_82 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr2)
{
    auto out_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp1 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp4 = tmp3.exp();
                        tmp4.store(out_ptr1 + static_cast<long>(x1 + (128L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = out_ptr2[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_84 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp13.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_relu_view_85 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_86 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                    auto tmp11 = static_cast<float>(1e-05);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    tmp19.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_88 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_89 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
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


cpp_fused_view_90 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_91 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                    auto tmp13 = static_cast<float>(1e-05);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_92 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_93 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_94 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr2)
{
    auto out_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp1 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp4 = tmp3.exp();
                        tmp4.store(out_ptr1 + static_cast<long>(x1 + (128L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = out_ptr2[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_view_95 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_96 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_relu_view_97 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                tmp1.store(out_ptr0 + static_cast<long>(x0));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp13.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_100 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_101 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                    auto tmp11 = static_cast<float>(1e-05);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    tmp19.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_104 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
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
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_106 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr2)
{
    auto out_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp1 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp4 = tmp3.exp();
                        tmp4.store(out_ptr1 + static_cast<long>(x1 + (128L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = out_ptr2[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                    auto tmp13 = static_cast<float>(1e-05);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_relu_view_109 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                tmp1.store(out_ptr0 + static_cast<long>(x0));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_113 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
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


cpp_fused_view_114 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp13.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_116 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    }
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
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_118 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr2)
{
    auto out_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp1 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp4 = tmp3.exp();
                        tmp4.store(out_ptr1 + static_cast<long>(x1 + (128L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = out_ptr2[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_view_119 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                    auto tmp11 = static_cast<float>(1e-05);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    tmp19.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_relu_view_121 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_122 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                    auto tmp13 = static_cast<float>(1e-05);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_124 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_125 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
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


cpp_fused_view_126 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_127 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_128 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_129 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_130 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr2)
{
    auto out_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp1 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp4 = tmp3.exp();
                        tmp4.store(out_ptr1 + static_cast<long>(x1 + (128L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = out_ptr2[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_view_131 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_132 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp13.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_relu_view_133 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_134 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                    auto tmp11 = static_cast<float>(1e-05);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    tmp19.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_136 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_137 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_139 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                    auto tmp13 = static_cast<float>(1e-05);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_140 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_141 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_142 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr2)
{
    auto out_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp1 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp4 = tmp3.exp();
                        tmp4.store(out_ptr1 + static_cast<long>(x1 + (128L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = out_ptr2[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_view_143 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_144 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_relu_view_145 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_146 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp13.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_147 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_148 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_149 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
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


cpp_fused_view_150 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_151 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                    auto tmp11 = static_cast<float>(1e-05);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    tmp19.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_152 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_153 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_154 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr2)
{
    auto out_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp1 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp4 = tmp3.exp();
                        tmp4.store(out_ptr1 + static_cast<long>(x1 + (128L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = out_ptr2[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_view_155 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_156 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                    auto tmp13 = static_cast<float>(1e-05);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_relu_view_157 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_158 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_159 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_160 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_161 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
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


cpp_fused_view_162 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_163 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp13.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_164 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
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
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_166 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr2)
{
    auto out_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp1 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp4 = tmp3.exp();
                        tmp4.store(out_ptr1 + static_cast<long>(x1 + (128L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = out_ptr2[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_view_167 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_168 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                    auto tmp11 = static_cast<float>(1e-05);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    tmp19.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_relu_view_169 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_170 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                    auto tmp13 = static_cast<float>(1e-05);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_171 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_173 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
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


cpp_fused_view_174 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_175 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_176 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    }
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
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_178 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr2)
{
    auto out_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp1 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp4 = tmp3.exp();
                        tmp4.store(out_ptr1 + static_cast<long>(x1 + (128L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = out_ptr2[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_view_179 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_180 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp13.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_relu_view_181 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_182 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                    auto tmp11 = static_cast<float>(1e-05);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    tmp19.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_183 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_184 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_185 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
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


cpp_fused_view_186 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_187 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                    auto tmp13 = static_cast<float>(1e-05);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_188 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_189 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_190 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr2)
{
    auto out_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp1 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp4 = tmp3.exp();
                        tmp4.store(out_ptr1 + static_cast<long>(x1 + (128L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = out_ptr2[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_view_191 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_192 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_relu_view_193 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_194 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp13.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_196 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_197 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
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


cpp_fused_view_198 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_199 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                    auto tmp11 = static_cast<float>(1e-05);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    tmp19.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_200 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_201 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_202 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr2)
{
    auto out_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp1 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp4 = tmp3.exp();
                        tmp4.store(out_ptr1 + static_cast<long>(x1 + (128L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = out_ptr2[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_view_203 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_204 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                    auto tmp13 = static_cast<float>(1e-05);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_relu_threshold_backward_view_205 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       bool* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(1L))
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
}
''')


cpp_fused_add_native_layer_norm_view_206 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_207 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_208 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_detach_209 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto in_ptr0 = in_out_ptr0;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
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


cpp_fused_view_210 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_211 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp13.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_212 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_213 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_214 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr2)
{
    auto out_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp1 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp4 = tmp3.exp();
                        tmp4.store(out_ptr1 + static_cast<long>(x1 + (128L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = out_ptr2[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_view_215 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_216 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                    auto tmp11 = static_cast<float>(1e-05);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    tmp19.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_relu_threshold_backward_view_217 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       bool* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(1L))
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
}
''')


cpp_fused_add_native_layer_norm_view_218 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                    auto tmp13 = static_cast<float>(1e-05);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__log_softmax__softmax_add_detach_embedding_masked_fill_mul_native_layer_norm_native_layer_norm_backward_nll_loss_forward_relu_threshold_backward_219 = async_compile.cpp('''
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
                       float* in_out_ptr72,
                       float* in_out_ptr73,
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
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       long* out_ptr3,
                       float* out_ptr5,
                       bool* out_ptr6,
                       bool* out_ptr7,
                       bool* out_ptr8,
                       bool* out_ptr9,
                       bool* out_ptr10,
                       bool* out_ptr11,
                       bool* out_ptr12,
                       bool* out_ptr13,
                       bool* out_ptr14,
                       bool* out_ptr15,
                       bool* out_ptr16,
                       bool* out_ptr17,
                       bool* out_ptr18,
                       bool* out_ptr19,
                       bool* out_ptr20,
                       bool* out_ptr21,
                       bool* out_ptr22,
                       bool* out_ptr23,
                       bool* out_ptr24,
                       bool* out_ptr25,
                       bool* out_ptr26,
                       bool* out_ptr27)
{
    auto out_ptr4 = in_out_ptr0;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128112L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128112L*x0)));
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp0);
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128112L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128112L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128112L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128112L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = std::log(tmp4);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp3 - tmp6;
                    tmp7.store(out_ptr2 + static_cast<long>(x1 + (128112L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                {
                    long tmp_acc0 = 0;
                    float tmp_acc1 = 0;
                    for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr1[static_cast<long>(x0)];
                        auto tmp1 = static_cast<long>(-100);
                        auto tmp2 = tmp0 != tmp1;
                        auto tmp3 = c10::convert<long>(tmp2);
                        auto tmp4 = static_cast<long>(0);
                        auto tmp5 = tmp2 ? tmp0 : tmp4;
                        auto tmp6 = decltype(tmp5)(tmp5 + 128112);
                        auto tmp7 = tmp5 < 0;
                        auto tmp8 = tmp7 ? tmp6 : tmp5;
                        TORCH_CHECK((0 <= tmp8) & (tmp8 < 128112L), "index out of bounds: 0 <= tmp8 < 128112L")
                        auto tmp9 = out_ptr2[static_cast<long>(tmp8 + (128112L*x0))];
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
                    auto tmp1 = static_cast<float>(1024.0);
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
                    auto tmp1 = static_cast<float>(1024.0);
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
                    auto tmp1 = static_cast<float>(1024.0);
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
                    auto tmp1 = static_cast<float>(1024.0);
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
                    auto tmp1 = static_cast<float>(1024.0);
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
                    auto tmp1 = static_cast<float>(1024.0);
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
                    auto tmp1 = static_cast<float>(1024.0);
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr3[static_cast<long>(x0)];
                auto tmp1 = tmp0 * (tmp0>0);
                auto tmp2 = static_cast<float>(0.0);
                auto tmp3 = tmp1 <= tmp2;
                out_ptr6[static_cast<long>(x0)] = tmp3;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr9 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1024.0);
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
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr10 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1024.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr11 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr11 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr12 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1024.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr5[static_cast<long>(x0)];
                auto tmp1 = tmp0 * (tmp0>0);
                auto tmp2 = static_cast<float>(0.0);
                auto tmp3 = tmp1 <= tmp2;
                out_ptr7[static_cast<long>(x0)] = tmp3;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr13 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1024.0);
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
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr14 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1024.0);
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr15 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = in_ptr6[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr15 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr16 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1024.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr7[static_cast<long>(x0)];
                auto tmp1 = tmp0 * (tmp0>0);
                auto tmp2 = static_cast<float>(0.0);
                auto tmp3 = tmp1 <= tmp2;
                out_ptr8[static_cast<long>(x0)] = tmp3;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr17 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1024.0);
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
                    auto tmp1 = static_cast<float>(1024.0);
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
                    auto tmp1 = in_ptr8[static_cast<long>(x0)];
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
                    auto tmp1 = static_cast<float>(1024.0);
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr9[static_cast<long>(x0)];
                auto tmp1 = tmp0 * (tmp0>0);
                auto tmp2 = static_cast<float>(0.0);
                auto tmp3 = tmp1 <= tmp2;
                out_ptr9[static_cast<long>(x0)] = tmp3;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr21 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1024.0);
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
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr22 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1024.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr23 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = in_ptr10[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr23 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr24 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1024.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr11[static_cast<long>(x0)];
                auto tmp1 = tmp0 * (tmp0>0);
                auto tmp2 = static_cast<float>(0.0);
                auto tmp3 = tmp1 <= tmp2;
                out_ptr10[static_cast<long>(x0)] = tmp3;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr25 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1024.0);
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
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr26 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1024.0);
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr27 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = in_ptr12[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr27 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr28 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1024.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr28 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr13[static_cast<long>(x0)];
                auto tmp1 = tmp0 * (tmp0>0);
                auto tmp2 = static_cast<float>(0.0);
                auto tmp3 = tmp1 <= tmp2;
                out_ptr11[static_cast<long>(x0)] = tmp3;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr29 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1024.0);
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
                    auto tmp1 = static_cast<float>(1024.0);
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
                    auto tmp1 = in_ptr14[static_cast<long>(x0)];
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
                    auto tmp1 = static_cast<float>(1024.0);
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr15[static_cast<long>(x0)];
                auto tmp1 = tmp0 * (tmp0>0);
                auto tmp2 = static_cast<float>(0.0);
                auto tmp3 = tmp1 <= tmp2;
                out_ptr12[static_cast<long>(x0)] = tmp3;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr33 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1024.0);
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
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr34 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1024.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr34 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr35 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = in_ptr16[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr35 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr36 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1024.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr17[static_cast<long>(x0)];
                auto tmp1 = tmp0 * (tmp0>0);
                auto tmp2 = static_cast<float>(0.0);
                auto tmp3 = tmp1 <= tmp2;
                out_ptr13[static_cast<long>(x0)] = tmp3;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr37 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1024.0);
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
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr38 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1024.0);
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr39 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = in_ptr18[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr39 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr40 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1024.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr40 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr19[static_cast<long>(x0)];
                auto tmp1 = tmp0 * (tmp0>0);
                auto tmp2 = static_cast<float>(0.0);
                auto tmp3 = tmp1 <= tmp2;
                out_ptr14[static_cast<long>(x0)] = tmp3;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr41 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1024.0);
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
                    auto tmp1 = static_cast<float>(1024.0);
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
                    auto tmp1 = in_ptr20[static_cast<long>(x0)];
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
                    auto tmp1 = static_cast<float>(1024.0);
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr21[static_cast<long>(x0)];
                auto tmp1 = tmp0 * (tmp0>0);
                auto tmp2 = static_cast<float>(0.0);
                auto tmp3 = tmp1 <= tmp2;
                out_ptr15[static_cast<long>(x0)] = tmp3;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr45 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1024.0);
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
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr46 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1024.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr46 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr47 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = in_ptr22[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr47 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr48 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1024.0);
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
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr49 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1024.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr23[static_cast<long>(x0)];
                auto tmp1 = tmp0 * (tmp0>0);
                auto tmp2 = static_cast<float>(0.0);
                auto tmp3 = tmp1 <= tmp2;
                out_ptr16[static_cast<long>(x0)] = tmp3;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr50 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1024.0);
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
                    auto tmp1 = static_cast<float>(1024.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr24[static_cast<long>(x0)];
                auto tmp1 = tmp0 * (tmp0>0);
                auto tmp2 = static_cast<float>(0.0);
                auto tmp3 = tmp1 <= tmp2;
                out_ptr17[static_cast<long>(x0)] = tmp3;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr52 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1024.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr52 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr53 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1024.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr25[static_cast<long>(x0)];
                auto tmp1 = tmp0 * (tmp0>0);
                auto tmp2 = static_cast<float>(0.0);
                auto tmp3 = tmp1 <= tmp2;
                out_ptr18[static_cast<long>(x0)] = tmp3;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr54 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1024.0);
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
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr55 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1024.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr55 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr26[static_cast<long>(x0)];
                auto tmp1 = tmp0 * (tmp0>0);
                auto tmp2 = static_cast<float>(0.0);
                auto tmp3 = tmp1 <= tmp2;
                out_ptr19[static_cast<long>(x0)] = tmp3;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr56 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1024.0);
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
                    auto tmp1 = static_cast<float>(1024.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr27[static_cast<long>(x0)];
                auto tmp1 = tmp0 * (tmp0>0);
                auto tmp2 = static_cast<float>(0.0);
                auto tmp3 = tmp1 <= tmp2;
                out_ptr20[static_cast<long>(x0)] = tmp3;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr58 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1024.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr58 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr59 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1024.0);
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr28[static_cast<long>(x0)];
                auto tmp1 = tmp0 * (tmp0>0);
                auto tmp2 = static_cast<float>(0.0);
                auto tmp3 = tmp1 <= tmp2;
                out_ptr21[static_cast<long>(x0)] = tmp3;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr60 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1024.0);
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
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr61 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1024.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr61 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr29[static_cast<long>(x0)];
                auto tmp1 = tmp0 * (tmp0>0);
                auto tmp2 = static_cast<float>(0.0);
                auto tmp3 = tmp1 <= tmp2;
                out_ptr22[static_cast<long>(x0)] = tmp3;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr62 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1024.0);
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
                    auto tmp1 = static_cast<float>(1024.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr30[static_cast<long>(x0)];
                auto tmp1 = tmp0 * (tmp0>0);
                auto tmp2 = static_cast<float>(0.0);
                auto tmp3 = tmp1 <= tmp2;
                out_ptr23[static_cast<long>(x0)] = tmp3;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr64 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1024.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr64 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr65 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1024.0);
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr31[static_cast<long>(x0)];
                auto tmp1 = tmp0 * (tmp0>0);
                auto tmp2 = static_cast<float>(0.0);
                auto tmp3 = tmp1 <= tmp2;
                out_ptr24[static_cast<long>(x0)] = tmp3;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr66 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1024.0);
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
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr67 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1024.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr67 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr32[static_cast<long>(x0)];
                auto tmp1 = tmp0 * (tmp0>0);
                auto tmp2 = static_cast<float>(0.0);
                auto tmp3 = tmp1 <= tmp2;
                out_ptr25[static_cast<long>(x0)] = tmp3;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr68 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1024.0);
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
                    auto tmp1 = static_cast<float>(1024.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr33[static_cast<long>(x0)];
                auto tmp1 = tmp0 * (tmp0>0);
                auto tmp2 = static_cast<float>(0.0);
                auto tmp3 = tmp1 <= tmp2;
                out_ptr26[static_cast<long>(x0)] = tmp3;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr70 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1024.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr70 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr71 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1024.0);
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr34[static_cast<long>(x0)];
                auto tmp1 = tmp0 * (tmp0>0);
                auto tmp2 = static_cast<float>(0.0);
                auto tmp3 = tmp1 <= tmp2;
                out_ptr27[static_cast<long>(x0)] = tmp3;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr72 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1024.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr72 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr73 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1024.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr73 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444, primals_445, primals_446, primals_447, primals_448, primals_449, primals_450, primals_451, primals_452, primals_453, primals_454, primals_455, primals_456, primals_457, primals_458, primals_459, primals_460, primals_461, primals_462, primals_463, primals_464, primals_465, primals_466, primals_467, primals_468, primals_469, primals_470, primals_471, primals_472, primals_473, primals_474, primals_475, primals_476, primals_477, primals_478, primals_479, primals_480, primals_481, primals_482, primals_483, primals_484, primals_485, primals_486, primals_487, primals_488, primals_489, primals_490, primals_491, primals_492, primals_493, primals_494, primals_495, primals_496, primals_497, primals_498, primals_499, primals_500, primals_501, primals_502, primals_503, primals_504, primals_505, primals_506, primals_507, primals_508, primals_509, primals_510, primals_511, primals_512, primals_513, primals_514, primals_515, primals_516 = args
    args.clear()
    assert_size_stride(primals_1, (128112, 1024), (1024, 1))
    assert_size_stride(primals_2, (1024, ), (1, ))
    assert_size_stride(primals_3, (1024, ), (1, ))
    assert_size_stride(primals_4, (1024, 1024), (1024, 1))
    assert_size_stride(primals_5, (1024, ), (1, ))
    assert_size_stride(primals_6, (1024, 1024), (1024, 1))
    assert_size_stride(primals_7, (1024, ), (1, ))
    assert_size_stride(primals_8, (1024, 1024), (1024, 1))
    assert_size_stride(primals_9, (1024, ), (1, ))
    assert_size_stride(primals_10, (1024, 1024), (1024, 1))
    assert_size_stride(primals_11, (1024, ), (1, ))
    assert_size_stride(primals_12, (1024, ), (1, ))
    assert_size_stride(primals_13, (1024, ), (1, ))
    assert_size_stride(primals_14, (4096, 1024), (1024, 1))
    assert_size_stride(primals_15, (4096, ), (1, ))
    assert_size_stride(primals_16, (1024, 4096), (4096, 1))
    assert_size_stride(primals_17, (1024, ), (1, ))
    assert_size_stride(primals_18, (1024, ), (1, ))
    assert_size_stride(primals_19, (1024, ), (1, ))
    assert_size_stride(primals_20, (1024, 1024), (1024, 1))
    assert_size_stride(primals_21, (1024, ), (1, ))
    assert_size_stride(primals_22, (1024, 1024), (1024, 1))
    assert_size_stride(primals_23, (1024, ), (1, ))
    assert_size_stride(primals_24, (1024, 1024), (1024, 1))
    assert_size_stride(primals_25, (1024, ), (1, ))
    assert_size_stride(primals_26, (1024, 1024), (1024, 1))
    assert_size_stride(primals_27, (1024, ), (1, ))
    assert_size_stride(primals_28, (1024, ), (1, ))
    assert_size_stride(primals_29, (1024, ), (1, ))
    assert_size_stride(primals_30, (4096, 1024), (1024, 1))
    assert_size_stride(primals_31, (4096, ), (1, ))
    assert_size_stride(primals_32, (1024, 4096), (4096, 1))
    assert_size_stride(primals_33, (1024, ), (1, ))
    assert_size_stride(primals_34, (1024, ), (1, ))
    assert_size_stride(primals_35, (1024, ), (1, ))
    assert_size_stride(primals_36, (1024, 1024), (1024, 1))
    assert_size_stride(primals_37, (1024, ), (1, ))
    assert_size_stride(primals_38, (1024, 1024), (1024, 1))
    assert_size_stride(primals_39, (1024, ), (1, ))
    assert_size_stride(primals_40, (1024, 1024), (1024, 1))
    assert_size_stride(primals_41, (1024, ), (1, ))
    assert_size_stride(primals_42, (1024, 1024), (1024, 1))
    assert_size_stride(primals_43, (1024, ), (1, ))
    assert_size_stride(primals_44, (1024, ), (1, ))
    assert_size_stride(primals_45, (1024, ), (1, ))
    assert_size_stride(primals_46, (4096, 1024), (1024, 1))
    assert_size_stride(primals_47, (4096, ), (1, ))
    assert_size_stride(primals_48, (1024, 4096), (4096, 1))
    assert_size_stride(primals_49, (1024, ), (1, ))
    assert_size_stride(primals_50, (1024, ), (1, ))
    assert_size_stride(primals_51, (1024, ), (1, ))
    assert_size_stride(primals_52, (1024, 1024), (1024, 1))
    assert_size_stride(primals_53, (1024, ), (1, ))
    assert_size_stride(primals_54, (1024, 1024), (1024, 1))
    assert_size_stride(primals_55, (1024, ), (1, ))
    assert_size_stride(primals_56, (1024, 1024), (1024, 1))
    assert_size_stride(primals_57, (1024, ), (1, ))
    assert_size_stride(primals_58, (1024, 1024), (1024, 1))
    assert_size_stride(primals_59, (1024, ), (1, ))
    assert_size_stride(primals_60, (1024, ), (1, ))
    assert_size_stride(primals_61, (1024, ), (1, ))
    assert_size_stride(primals_62, (4096, 1024), (1024, 1))
    assert_size_stride(primals_63, (4096, ), (1, ))
    assert_size_stride(primals_64, (1024, 4096), (4096, 1))
    assert_size_stride(primals_65, (1024, ), (1, ))
    assert_size_stride(primals_66, (1024, ), (1, ))
    assert_size_stride(primals_67, (1024, ), (1, ))
    assert_size_stride(primals_68, (1024, 1024), (1024, 1))
    assert_size_stride(primals_69, (1024, ), (1, ))
    assert_size_stride(primals_70, (1024, 1024), (1024, 1))
    assert_size_stride(primals_71, (1024, ), (1, ))
    assert_size_stride(primals_72, (1024, 1024), (1024, 1))
    assert_size_stride(primals_73, (1024, ), (1, ))
    assert_size_stride(primals_74, (1024, 1024), (1024, 1))
    assert_size_stride(primals_75, (1024, ), (1, ))
    assert_size_stride(primals_76, (1024, ), (1, ))
    assert_size_stride(primals_77, (1024, ), (1, ))
    assert_size_stride(primals_78, (4096, 1024), (1024, 1))
    assert_size_stride(primals_79, (4096, ), (1, ))
    assert_size_stride(primals_80, (1024, 4096), (4096, 1))
    assert_size_stride(primals_81, (1024, ), (1, ))
    assert_size_stride(primals_82, (1024, ), (1, ))
    assert_size_stride(primals_83, (1024, ), (1, ))
    assert_size_stride(primals_84, (1024, 1024), (1024, 1))
    assert_size_stride(primals_85, (1024, ), (1, ))
    assert_size_stride(primals_86, (1024, 1024), (1024, 1))
    assert_size_stride(primals_87, (1024, ), (1, ))
    assert_size_stride(primals_88, (1024, 1024), (1024, 1))
    assert_size_stride(primals_89, (1024, ), (1, ))
    assert_size_stride(primals_90, (1024, 1024), (1024, 1))
    assert_size_stride(primals_91, (1024, ), (1, ))
    assert_size_stride(primals_92, (1024, ), (1, ))
    assert_size_stride(primals_93, (1024, ), (1, ))
    assert_size_stride(primals_94, (4096, 1024), (1024, 1))
    assert_size_stride(primals_95, (4096, ), (1, ))
    assert_size_stride(primals_96, (1024, 4096), (4096, 1))
    assert_size_stride(primals_97, (1024, ), (1, ))
    assert_size_stride(primals_98, (1024, ), (1, ))
    assert_size_stride(primals_99, (1024, ), (1, ))
    assert_size_stride(primals_100, (1024, 1024), (1024, 1))
    assert_size_stride(primals_101, (1024, ), (1, ))
    assert_size_stride(primals_102, (1024, 1024), (1024, 1))
    assert_size_stride(primals_103, (1024, ), (1, ))
    assert_size_stride(primals_104, (1024, 1024), (1024, 1))
    assert_size_stride(primals_105, (1024, ), (1, ))
    assert_size_stride(primals_106, (1024, 1024), (1024, 1))
    assert_size_stride(primals_107, (1024, ), (1, ))
    assert_size_stride(primals_108, (1024, ), (1, ))
    assert_size_stride(primals_109, (1024, ), (1, ))
    assert_size_stride(primals_110, (4096, 1024), (1024, 1))
    assert_size_stride(primals_111, (4096, ), (1, ))
    assert_size_stride(primals_112, (1024, 4096), (4096, 1))
    assert_size_stride(primals_113, (1024, ), (1, ))
    assert_size_stride(primals_114, (1024, ), (1, ))
    assert_size_stride(primals_115, (1024, ), (1, ))
    assert_size_stride(primals_116, (1024, 1024), (1024, 1))
    assert_size_stride(primals_117, (1024, ), (1, ))
    assert_size_stride(primals_118, (1024, 1024), (1024, 1))
    assert_size_stride(primals_119, (1024, ), (1, ))
    assert_size_stride(primals_120, (1024, 1024), (1024, 1))
    assert_size_stride(primals_121, (1024, ), (1, ))
    assert_size_stride(primals_122, (1024, 1024), (1024, 1))
    assert_size_stride(primals_123, (1024, ), (1, ))
    assert_size_stride(primals_124, (1024, ), (1, ))
    assert_size_stride(primals_125, (1024, ), (1, ))
    assert_size_stride(primals_126, (4096, 1024), (1024, 1))
    assert_size_stride(primals_127, (4096, ), (1, ))
    assert_size_stride(primals_128, (1024, 4096), (4096, 1))
    assert_size_stride(primals_129, (1024, ), (1, ))
    assert_size_stride(primals_130, (1024, ), (1, ))
    assert_size_stride(primals_131, (1024, ), (1, ))
    assert_size_stride(primals_132, (1024, 1024), (1024, 1))
    assert_size_stride(primals_133, (1024, ), (1, ))
    assert_size_stride(primals_134, (1024, 1024), (1024, 1))
    assert_size_stride(primals_135, (1024, ), (1, ))
    assert_size_stride(primals_136, (1024, 1024), (1024, 1))
    assert_size_stride(primals_137, (1024, ), (1, ))
    assert_size_stride(primals_138, (1024, 1024), (1024, 1))
    assert_size_stride(primals_139, (1024, ), (1, ))
    assert_size_stride(primals_140, (1024, ), (1, ))
    assert_size_stride(primals_141, (1024, ), (1, ))
    assert_size_stride(primals_142, (4096, 1024), (1024, 1))
    assert_size_stride(primals_143, (4096, ), (1, ))
    assert_size_stride(primals_144, (1024, 4096), (4096, 1))
    assert_size_stride(primals_145, (1024, ), (1, ))
    assert_size_stride(primals_146, (1024, ), (1, ))
    assert_size_stride(primals_147, (1024, ), (1, ))
    assert_size_stride(primals_148, (1024, 1024), (1024, 1))
    assert_size_stride(primals_149, (1024, ), (1, ))
    assert_size_stride(primals_150, (1024, 1024), (1024, 1))
    assert_size_stride(primals_151, (1024, ), (1, ))
    assert_size_stride(primals_152, (1024, 1024), (1024, 1))
    assert_size_stride(primals_153, (1024, ), (1, ))
    assert_size_stride(primals_154, (1024, 1024), (1024, 1))
    assert_size_stride(primals_155, (1024, ), (1, ))
    assert_size_stride(primals_156, (1024, ), (1, ))
    assert_size_stride(primals_157, (1024, ), (1, ))
    assert_size_stride(primals_158, (4096, 1024), (1024, 1))
    assert_size_stride(primals_159, (4096, ), (1, ))
    assert_size_stride(primals_160, (1024, 4096), (4096, 1))
    assert_size_stride(primals_161, (1024, ), (1, ))
    assert_size_stride(primals_162, (1024, ), (1, ))
    assert_size_stride(primals_163, (1024, ), (1, ))
    assert_size_stride(primals_164, (1024, 1024), (1024, 1))
    assert_size_stride(primals_165, (1024, ), (1, ))
    assert_size_stride(primals_166, (1024, 1024), (1024, 1))
    assert_size_stride(primals_167, (1024, ), (1, ))
    assert_size_stride(primals_168, (1024, 1024), (1024, 1))
    assert_size_stride(primals_169, (1024, ), (1, ))
    assert_size_stride(primals_170, (1024, 1024), (1024, 1))
    assert_size_stride(primals_171, (1024, ), (1, ))
    assert_size_stride(primals_172, (1024, ), (1, ))
    assert_size_stride(primals_173, (1024, ), (1, ))
    assert_size_stride(primals_174, (4096, 1024), (1024, 1))
    assert_size_stride(primals_175, (4096, ), (1, ))
    assert_size_stride(primals_176, (1024, 4096), (4096, 1))
    assert_size_stride(primals_177, (1024, ), (1, ))
    assert_size_stride(primals_178, (1024, ), (1, ))
    assert_size_stride(primals_179, (1024, ), (1, ))
    assert_size_stride(primals_180, (1024, 1024), (1024, 1))
    assert_size_stride(primals_181, (1024, ), (1, ))
    assert_size_stride(primals_182, (1024, 1024), (1024, 1))
    assert_size_stride(primals_183, (1024, ), (1, ))
    assert_size_stride(primals_184, (1024, 1024), (1024, 1))
    assert_size_stride(primals_185, (1024, ), (1, ))
    assert_size_stride(primals_186, (1024, 1024), (1024, 1))
    assert_size_stride(primals_187, (1024, ), (1, ))
    assert_size_stride(primals_188, (1024, ), (1, ))
    assert_size_stride(primals_189, (1024, ), (1, ))
    assert_size_stride(primals_190, (4096, 1024), (1024, 1))
    assert_size_stride(primals_191, (4096, ), (1, ))
    assert_size_stride(primals_192, (1024, 4096), (4096, 1))
    assert_size_stride(primals_193, (1024, ), (1, ))
    assert_size_stride(primals_194, (1024, ), (1, ))
    assert_size_stride(primals_195, (1024, ), (1, ))
    assert_size_stride(primals_196, (128112, 1024), (1024, 1))
    assert_size_stride(primals_197, (1024, ), (1, ))
    assert_size_stride(primals_198, (1024, ), (1, ))
    assert_size_stride(primals_199, (1024, 1024), (1024, 1))
    assert_size_stride(primals_200, (1024, ), (1, ))
    assert_size_stride(primals_201, (1024, 1024), (1024, 1))
    assert_size_stride(primals_202, (1024, ), (1, ))
    assert_size_stride(primals_203, (1024, 1024), (1024, 1))
    assert_size_stride(primals_204, (1024, ), (1, ))
    assert_size_stride(primals_205, (1024, 1024), (1024, 1))
    assert_size_stride(primals_206, (1024, ), (1, ))
    assert_size_stride(primals_207, (1024, ), (1, ))
    assert_size_stride(primals_208, (1024, ), (1, ))
    assert_size_stride(primals_209, (1024, 1024), (1024, 1))
    assert_size_stride(primals_210, (1024, ), (1, ))
    assert_size_stride(primals_211, (1024, 1024), (1024, 1))
    assert_size_stride(primals_212, (1024, ), (1, ))
    assert_size_stride(primals_213, (1024, 1024), (1024, 1))
    assert_size_stride(primals_214, (1024, ), (1, ))
    assert_size_stride(primals_215, (1024, 1024), (1024, 1))
    assert_size_stride(primals_216, (1024, ), (1, ))
    assert_size_stride(primals_217, (1024, ), (1, ))
    assert_size_stride(primals_218, (1024, ), (1, ))
    assert_size_stride(primals_219, (4096, 1024), (1024, 1))
    assert_size_stride(primals_220, (4096, ), (1, ))
    assert_size_stride(primals_221, (1024, 4096), (4096, 1))
    assert_size_stride(primals_222, (1024, ), (1, ))
    assert_size_stride(primals_223, (1024, ), (1, ))
    assert_size_stride(primals_224, (1024, ), (1, ))
    assert_size_stride(primals_225, (1024, 1024), (1024, 1))
    assert_size_stride(primals_226, (1024, ), (1, ))
    assert_size_stride(primals_227, (1024, 1024), (1024, 1))
    assert_size_stride(primals_228, (1024, ), (1, ))
    assert_size_stride(primals_229, (1024, 1024), (1024, 1))
    assert_size_stride(primals_230, (1024, ), (1, ))
    assert_size_stride(primals_231, (1024, 1024), (1024, 1))
    assert_size_stride(primals_232, (1024, ), (1, ))
    assert_size_stride(primals_233, (1024, ), (1, ))
    assert_size_stride(primals_234, (1024, ), (1, ))
    assert_size_stride(primals_235, (1024, 1024), (1024, 1))
    assert_size_stride(primals_236, (1024, ), (1, ))
    assert_size_stride(primals_237, (1024, 1024), (1024, 1))
    assert_size_stride(primals_238, (1024, ), (1, ))
    assert_size_stride(primals_239, (1024, 1024), (1024, 1))
    assert_size_stride(primals_240, (1024, ), (1, ))
    assert_size_stride(primals_241, (1024, 1024), (1024, 1))
    assert_size_stride(primals_242, (1024, ), (1, ))
    assert_size_stride(primals_243, (1024, ), (1, ))
    assert_size_stride(primals_244, (1024, ), (1, ))
    assert_size_stride(primals_245, (4096, 1024), (1024, 1))
    assert_size_stride(primals_246, (4096, ), (1, ))
    assert_size_stride(primals_247, (1024, 4096), (4096, 1))
    assert_size_stride(primals_248, (1024, ), (1, ))
    assert_size_stride(primals_249, (1024, ), (1, ))
    assert_size_stride(primals_250, (1024, ), (1, ))
    assert_size_stride(primals_251, (1024, 1024), (1024, 1))
    assert_size_stride(primals_252, (1024, ), (1, ))
    assert_size_stride(primals_253, (1024, 1024), (1024, 1))
    assert_size_stride(primals_254, (1024, ), (1, ))
    assert_size_stride(primals_255, (1024, 1024), (1024, 1))
    assert_size_stride(primals_256, (1024, ), (1, ))
    assert_size_stride(primals_257, (1024, 1024), (1024, 1))
    assert_size_stride(primals_258, (1024, ), (1, ))
    assert_size_stride(primals_259, (1024, ), (1, ))
    assert_size_stride(primals_260, (1024, ), (1, ))
    assert_size_stride(primals_261, (1024, 1024), (1024, 1))
    assert_size_stride(primals_262, (1024, ), (1, ))
    assert_size_stride(primals_263, (1024, 1024), (1024, 1))
    assert_size_stride(primals_264, (1024, ), (1, ))
    assert_size_stride(primals_265, (1024, 1024), (1024, 1))
    assert_size_stride(primals_266, (1024, ), (1, ))
    assert_size_stride(primals_267, (1024, 1024), (1024, 1))
    assert_size_stride(primals_268, (1024, ), (1, ))
    assert_size_stride(primals_269, (1024, ), (1, ))
    assert_size_stride(primals_270, (1024, ), (1, ))
    assert_size_stride(primals_271, (4096, 1024), (1024, 1))
    assert_size_stride(primals_272, (4096, ), (1, ))
    assert_size_stride(primals_273, (1024, 4096), (4096, 1))
    assert_size_stride(primals_274, (1024, ), (1, ))
    assert_size_stride(primals_275, (1024, ), (1, ))
    assert_size_stride(primals_276, (1024, ), (1, ))
    assert_size_stride(primals_277, (1024, 1024), (1024, 1))
    assert_size_stride(primals_278, (1024, ), (1, ))
    assert_size_stride(primals_279, (1024, 1024), (1024, 1))
    assert_size_stride(primals_280, (1024, ), (1, ))
    assert_size_stride(primals_281, (1024, 1024), (1024, 1))
    assert_size_stride(primals_282, (1024, ), (1, ))
    assert_size_stride(primals_283, (1024, 1024), (1024, 1))
    assert_size_stride(primals_284, (1024, ), (1, ))
    assert_size_stride(primals_285, (1024, ), (1, ))
    assert_size_stride(primals_286, (1024, ), (1, ))
    assert_size_stride(primals_287, (1024, 1024), (1024, 1))
    assert_size_stride(primals_288, (1024, ), (1, ))
    assert_size_stride(primals_289, (1024, 1024), (1024, 1))
    assert_size_stride(primals_290, (1024, ), (1, ))
    assert_size_stride(primals_291, (1024, 1024), (1024, 1))
    assert_size_stride(primals_292, (1024, ), (1, ))
    assert_size_stride(primals_293, (1024, 1024), (1024, 1))
    assert_size_stride(primals_294, (1024, ), (1, ))
    assert_size_stride(primals_295, (1024, ), (1, ))
    assert_size_stride(primals_296, (1024, ), (1, ))
    assert_size_stride(primals_297, (4096, 1024), (1024, 1))
    assert_size_stride(primals_298, (4096, ), (1, ))
    assert_size_stride(primals_299, (1024, 4096), (4096, 1))
    assert_size_stride(primals_300, (1024, ), (1, ))
    assert_size_stride(primals_301, (1024, ), (1, ))
    assert_size_stride(primals_302, (1024, ), (1, ))
    assert_size_stride(primals_303, (1024, 1024), (1024, 1))
    assert_size_stride(primals_304, (1024, ), (1, ))
    assert_size_stride(primals_305, (1024, 1024), (1024, 1))
    assert_size_stride(primals_306, (1024, ), (1, ))
    assert_size_stride(primals_307, (1024, 1024), (1024, 1))
    assert_size_stride(primals_308, (1024, ), (1, ))
    assert_size_stride(primals_309, (1024, 1024), (1024, 1))
    assert_size_stride(primals_310, (1024, ), (1, ))
    assert_size_stride(primals_311, (1024, ), (1, ))
    assert_size_stride(primals_312, (1024, ), (1, ))
    assert_size_stride(primals_313, (1024, 1024), (1024, 1))
    assert_size_stride(primals_314, (1024, ), (1, ))
    assert_size_stride(primals_315, (1024, 1024), (1024, 1))
    assert_size_stride(primals_316, (1024, ), (1, ))
    assert_size_stride(primals_317, (1024, 1024), (1024, 1))
    assert_size_stride(primals_318, (1024, ), (1, ))
    assert_size_stride(primals_319, (1024, 1024), (1024, 1))
    assert_size_stride(primals_320, (1024, ), (1, ))
    assert_size_stride(primals_321, (1024, ), (1, ))
    assert_size_stride(primals_322, (1024, ), (1, ))
    assert_size_stride(primals_323, (4096, 1024), (1024, 1))
    assert_size_stride(primals_324, (4096, ), (1, ))
    assert_size_stride(primals_325, (1024, 4096), (4096, 1))
    assert_size_stride(primals_326, (1024, ), (1, ))
    assert_size_stride(primals_327, (1024, ), (1, ))
    assert_size_stride(primals_328, (1024, ), (1, ))
    assert_size_stride(primals_329, (1024, 1024), (1024, 1))
    assert_size_stride(primals_330, (1024, ), (1, ))
    assert_size_stride(primals_331, (1024, 1024), (1024, 1))
    assert_size_stride(primals_332, (1024, ), (1, ))
    assert_size_stride(primals_333, (1024, 1024), (1024, 1))
    assert_size_stride(primals_334, (1024, ), (1, ))
    assert_size_stride(primals_335, (1024, 1024), (1024, 1))
    assert_size_stride(primals_336, (1024, ), (1, ))
    assert_size_stride(primals_337, (1024, ), (1, ))
    assert_size_stride(primals_338, (1024, ), (1, ))
    assert_size_stride(primals_339, (1024, 1024), (1024, 1))
    assert_size_stride(primals_340, (1024, ), (1, ))
    assert_size_stride(primals_341, (1024, 1024), (1024, 1))
    assert_size_stride(primals_342, (1024, ), (1, ))
    assert_size_stride(primals_343, (1024, 1024), (1024, 1))
    assert_size_stride(primals_344, (1024, ), (1, ))
    assert_size_stride(primals_345, (1024, 1024), (1024, 1))
    assert_size_stride(primals_346, (1024, ), (1, ))
    assert_size_stride(primals_347, (1024, ), (1, ))
    assert_size_stride(primals_348, (1024, ), (1, ))
    assert_size_stride(primals_349, (4096, 1024), (1024, 1))
    assert_size_stride(primals_350, (4096, ), (1, ))
    assert_size_stride(primals_351, (1024, 4096), (4096, 1))
    assert_size_stride(primals_352, (1024, ), (1, ))
    assert_size_stride(primals_353, (1024, ), (1, ))
    assert_size_stride(primals_354, (1024, ), (1, ))
    assert_size_stride(primals_355, (1024, 1024), (1024, 1))
    assert_size_stride(primals_356, (1024, ), (1, ))
    assert_size_stride(primals_357, (1024, 1024), (1024, 1))
    assert_size_stride(primals_358, (1024, ), (1, ))
    assert_size_stride(primals_359, (1024, 1024), (1024, 1))
    assert_size_stride(primals_360, (1024, ), (1, ))
    assert_size_stride(primals_361, (1024, 1024), (1024, 1))
    assert_size_stride(primals_362, (1024, ), (1, ))
    assert_size_stride(primals_363, (1024, ), (1, ))
    assert_size_stride(primals_364, (1024, ), (1, ))
    assert_size_stride(primals_365, (1024, 1024), (1024, 1))
    assert_size_stride(primals_366, (1024, ), (1, ))
    assert_size_stride(primals_367, (1024, 1024), (1024, 1))
    assert_size_stride(primals_368, (1024, ), (1, ))
    assert_size_stride(primals_369, (1024, 1024), (1024, 1))
    assert_size_stride(primals_370, (1024, ), (1, ))
    assert_size_stride(primals_371, (1024, 1024), (1024, 1))
    assert_size_stride(primals_372, (1024, ), (1, ))
    assert_size_stride(primals_373, (1024, ), (1, ))
    assert_size_stride(primals_374, (1024, ), (1, ))
    assert_size_stride(primals_375, (4096, 1024), (1024, 1))
    assert_size_stride(primals_376, (4096, ), (1, ))
    assert_size_stride(primals_377, (1024, 4096), (4096, 1))
    assert_size_stride(primals_378, (1024, ), (1, ))
    assert_size_stride(primals_379, (1024, ), (1, ))
    assert_size_stride(primals_380, (1024, ), (1, ))
    assert_size_stride(primals_381, (1024, 1024), (1024, 1))
    assert_size_stride(primals_382, (1024, ), (1, ))
    assert_size_stride(primals_383, (1024, 1024), (1024, 1))
    assert_size_stride(primals_384, (1024, ), (1, ))
    assert_size_stride(primals_385, (1024, 1024), (1024, 1))
    assert_size_stride(primals_386, (1024, ), (1, ))
    assert_size_stride(primals_387, (1024, 1024), (1024, 1))
    assert_size_stride(primals_388, (1024, ), (1, ))
    assert_size_stride(primals_389, (1024, ), (1, ))
    assert_size_stride(primals_390, (1024, ), (1, ))
    assert_size_stride(primals_391, (1024, 1024), (1024, 1))
    assert_size_stride(primals_392, (1024, ), (1, ))
    assert_size_stride(primals_393, (1024, 1024), (1024, 1))
    assert_size_stride(primals_394, (1024, ), (1, ))
    assert_size_stride(primals_395, (1024, 1024), (1024, 1))
    assert_size_stride(primals_396, (1024, ), (1, ))
    assert_size_stride(primals_397, (1024, 1024), (1024, 1))
    assert_size_stride(primals_398, (1024, ), (1, ))
    assert_size_stride(primals_399, (1024, ), (1, ))
    assert_size_stride(primals_400, (1024, ), (1, ))
    assert_size_stride(primals_401, (4096, 1024), (1024, 1))
    assert_size_stride(primals_402, (4096, ), (1, ))
    assert_size_stride(primals_403, (1024, 4096), (4096, 1))
    assert_size_stride(primals_404, (1024, ), (1, ))
    assert_size_stride(primals_405, (1024, ), (1, ))
    assert_size_stride(primals_406, (1024, ), (1, ))
    assert_size_stride(primals_407, (1024, 1024), (1024, 1))
    assert_size_stride(primals_408, (1024, ), (1, ))
    assert_size_stride(primals_409, (1024, 1024), (1024, 1))
    assert_size_stride(primals_410, (1024, ), (1, ))
    assert_size_stride(primals_411, (1024, 1024), (1024, 1))
    assert_size_stride(primals_412, (1024, ), (1, ))
    assert_size_stride(primals_413, (1024, 1024), (1024, 1))
    assert_size_stride(primals_414, (1024, ), (1, ))
    assert_size_stride(primals_415, (1024, ), (1, ))
    assert_size_stride(primals_416, (1024, ), (1, ))
    assert_size_stride(primals_417, (1024, 1024), (1024, 1))
    assert_size_stride(primals_418, (1024, ), (1, ))
    assert_size_stride(primals_419, (1024, 1024), (1024, 1))
    assert_size_stride(primals_420, (1024, ), (1, ))
    assert_size_stride(primals_421, (1024, 1024), (1024, 1))
    assert_size_stride(primals_422, (1024, ), (1, ))
    assert_size_stride(primals_423, (1024, 1024), (1024, 1))
    assert_size_stride(primals_424, (1024, ), (1, ))
    assert_size_stride(primals_425, (1024, ), (1, ))
    assert_size_stride(primals_426, (1024, ), (1, ))
    assert_size_stride(primals_427, (4096, 1024), (1024, 1))
    assert_size_stride(primals_428, (4096, ), (1, ))
    assert_size_stride(primals_429, (1024, 4096), (4096, 1))
    assert_size_stride(primals_430, (1024, ), (1, ))
    assert_size_stride(primals_431, (1024, ), (1, ))
    assert_size_stride(primals_432, (1024, ), (1, ))
    assert_size_stride(primals_433, (1024, 1024), (1024, 1))
    assert_size_stride(primals_434, (1024, ), (1, ))
    assert_size_stride(primals_435, (1024, 1024), (1024, 1))
    assert_size_stride(primals_436, (1024, ), (1, ))
    assert_size_stride(primals_437, (1024, 1024), (1024, 1))
    assert_size_stride(primals_438, (1024, ), (1, ))
    assert_size_stride(primals_439, (1024, 1024), (1024, 1))
    assert_size_stride(primals_440, (1024, ), (1, ))
    assert_size_stride(primals_441, (1024, ), (1, ))
    assert_size_stride(primals_442, (1024, ), (1, ))
    assert_size_stride(primals_443, (1024, 1024), (1024, 1))
    assert_size_stride(primals_444, (1024, ), (1, ))
    assert_size_stride(primals_445, (1024, 1024), (1024, 1))
    assert_size_stride(primals_446, (1024, ), (1, ))
    assert_size_stride(primals_447, (1024, 1024), (1024, 1))
    assert_size_stride(primals_448, (1024, ), (1, ))
    assert_size_stride(primals_449, (1024, 1024), (1024, 1))
    assert_size_stride(primals_450, (1024, ), (1, ))
    assert_size_stride(primals_451, (1024, ), (1, ))
    assert_size_stride(primals_452, (1024, ), (1, ))
    assert_size_stride(primals_453, (4096, 1024), (1024, 1))
    assert_size_stride(primals_454, (4096, ), (1, ))
    assert_size_stride(primals_455, (1024, 4096), (4096, 1))
    assert_size_stride(primals_456, (1024, ), (1, ))
    assert_size_stride(primals_457, (1024, ), (1, ))
    assert_size_stride(primals_458, (1024, ), (1, ))
    assert_size_stride(primals_459, (1024, 1024), (1024, 1))
    assert_size_stride(primals_460, (1024, ), (1, ))
    assert_size_stride(primals_461, (1024, 1024), (1024, 1))
    assert_size_stride(primals_462, (1024, ), (1, ))
    assert_size_stride(primals_463, (1024, 1024), (1024, 1))
    assert_size_stride(primals_464, (1024, ), (1, ))
    assert_size_stride(primals_465, (1024, 1024), (1024, 1))
    assert_size_stride(primals_466, (1024, ), (1, ))
    assert_size_stride(primals_467, (1024, ), (1, ))
    assert_size_stride(primals_468, (1024, ), (1, ))
    assert_size_stride(primals_469, (1024, 1024), (1024, 1))
    assert_size_stride(primals_470, (1024, ), (1, ))
    assert_size_stride(primals_471, (1024, 1024), (1024, 1))
    assert_size_stride(primals_472, (1024, ), (1, ))
    assert_size_stride(primals_473, (1024, 1024), (1024, 1))
    assert_size_stride(primals_474, (1024, ), (1, ))
    assert_size_stride(primals_475, (1024, 1024), (1024, 1))
    assert_size_stride(primals_476, (1024, ), (1, ))
    assert_size_stride(primals_477, (1024, ), (1, ))
    assert_size_stride(primals_478, (1024, ), (1, ))
    assert_size_stride(primals_479, (4096, 1024), (1024, 1))
    assert_size_stride(primals_480, (4096, ), (1, ))
    assert_size_stride(primals_481, (1024, 4096), (4096, 1))
    assert_size_stride(primals_482, (1024, ), (1, ))
    assert_size_stride(primals_483, (1024, ), (1, ))
    assert_size_stride(primals_484, (1024, ), (1, ))
    assert_size_stride(primals_485, (1024, 1024), (1024, 1))
    assert_size_stride(primals_486, (1024, ), (1, ))
    assert_size_stride(primals_487, (1024, 1024), (1024, 1))
    assert_size_stride(primals_488, (1024, ), (1, ))
    assert_size_stride(primals_489, (1024, 1024), (1024, 1))
    assert_size_stride(primals_490, (1024, ), (1, ))
    assert_size_stride(primals_491, (1024, 1024), (1024, 1))
    assert_size_stride(primals_492, (1024, ), (1, ))
    assert_size_stride(primals_493, (1024, ), (1, ))
    assert_size_stride(primals_494, (1024, ), (1, ))
    assert_size_stride(primals_495, (1024, 1024), (1024, 1))
    assert_size_stride(primals_496, (1024, ), (1, ))
    assert_size_stride(primals_497, (1024, 1024), (1024, 1))
    assert_size_stride(primals_498, (1024, ), (1, ))
    assert_size_stride(primals_499, (1024, 1024), (1024, 1))
    assert_size_stride(primals_500, (1024, ), (1, ))
    assert_size_stride(primals_501, (1024, 1024), (1024, 1))
    assert_size_stride(primals_502, (1024, ), (1, ))
    assert_size_stride(primals_503, (1024, ), (1, ))
    assert_size_stride(primals_504, (1024, ), (1, ))
    assert_size_stride(primals_505, (4096, 1024), (1024, 1))
    assert_size_stride(primals_506, (4096, ), (1, ))
    assert_size_stride(primals_507, (1024, 4096), (4096, 1))
    assert_size_stride(primals_508, (1024, ), (1, ))
    assert_size_stride(primals_509, (1024, ), (1, ))
    assert_size_stride(primals_510, (1024, ), (1, ))
    assert_size_stride(primals_511, (128112, 1024), (1024, 1))
    assert_size_stride(primals_512, (1026, 1024), (1024, 1))
    assert_size_stride(primals_513, (1026, 1024), (1024, 1))
    assert_size_stride(primals_514, (1, 128), (128, 1))
    assert_size_stride(primals_515, (1, 128), (128, 1))
    assert_size_stride(primals_516, (1, 128), (128, 1))
    buf0 = empty((1, 128), device='cpu', dtype=torch.int32)
    cpp_fused__to_copy_cumsum_ne_0(c_void_p(primals_516.data_ptr()), c_void_p(buf0.data_ptr()))
    # Source Nodes: [cumsum, mask, ne], Original ATen: [aten._to_copy, aten.cumsum, aten.ne]
    buf1 = aten.cumsum(buf0, 1)
    buf2 = buf1
    del buf1
    buf3 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf4 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf6 = empty((1, 128, 1024), device='cpu', dtype=torch.float32)
    buf7 = empty((128, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_embedding_mul_native_layer_norm_view_1(c_void_p(primals_516.data_ptr()), c_void_p(primals_1.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(primals_512.data_ptr()), c_void_p(primals_2.data_ptr()), c_void_p(primals_3.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(buf6.data_ptr()), c_void_p(buf7.data_ptr()))
    del primals_3
    buf8 = empty((128, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_encoder_layers_0_self_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_5, buf7, reinterpret_tensor(primals_4, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf8)
    del primals_5
    buf9 = empty((128, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_encoder_layers_0_self_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_7, buf7, reinterpret_tensor(primals_6, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf9)
    del primals_7
    buf10 = empty((128, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_encoder_layers_0_self_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_9, buf7, reinterpret_tensor(primals_8, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf10)
    del primals_9
    buf11 = empty((1, 16, 128, 64), device='cpu', dtype=torch.float32)
    buf12 = empty((1, 16, 128, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_2(c_void_p(buf8.data_ptr()), c_void_p(buf9.data_ptr()), c_void_p(buf11.data_ptr()), c_void_p(buf12.data_ptr()))
    buf13 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_weights], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf11, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf12, (16, 64, 128), (8192, 1, 64), 0), out=buf13)
    buf14 = empty((16, 128, 1), device='cpu', dtype=torch.float32)
    buf15 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    buf16 = empty((16, 128, 1), device='cpu', dtype=torch.float32)
    buf17 = buf15; del buf15  # reuse
    buf18 = reinterpret_tensor(buf9, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf9  # reuse
    cpp_fused__softmax_clone_3(c_void_p(buf17.data_ptr()), c_void_p(buf13.data_ptr()), c_void_p(buf10.data_ptr()), c_void_p(buf14.data_ptr()), c_void_p(buf16.data_ptr()), c_void_p(buf18.data_ptr()))
    buf19 = reinterpret_tensor(buf10, (16, 128, 64), (8192, 64, 1), 0); del buf10  # reuse
    # Source Nodes: [attn_output], Original ATen: [aten.bmm]
    extern_kernels.bmm(buf17, reinterpret_tensor(buf18, (16, 128, 64), (8192, 64, 1), 0), out=buf19)
    buf20 = buf8; del buf8  # reuse
    cpp_fused_view_4(c_void_p(buf19.data_ptr()), c_void_p(buf20.data_ptr()))
    buf21 = reinterpret_tensor(buf19, (128, 1024), (1024, 1), 0); del buf19  # reuse
    # Source Nodes: [hidden_states_3], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_11, buf20, reinterpret_tensor(primals_10, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf21)
    del primals_11
    buf22 = reinterpret_tensor(buf21, (1, 128, 1024), (131072, 1024, 1), 0); del buf21  # reuse
    buf23 = buf3; del buf3  # reuse
    buf24 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf26 = empty((1, 128, 1024), device='cpu', dtype=torch.float32)
    buf27 = empty((128, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_embedding_mul_native_layer_norm_view_5(c_void_p(buf22.data_ptr()), c_void_p(primals_516.data_ptr()), c_void_p(primals_1.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(primals_512.data_ptr()), c_void_p(primals_12.data_ptr()), c_void_p(primals_13.data_ptr()), c_void_p(buf23.data_ptr()), c_void_p(buf24.data_ptr()), c_void_p(buf26.data_ptr()), c_void_p(buf27.data_ptr()))
    del buf2
    del primals_1
    del primals_13
    del primals_512
    buf28 = empty((128, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_encoder_layers_0_fc1], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_15, buf27, reinterpret_tensor(primals_14, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf28)
    del primals_15
    buf29 = empty((128, 4096), device='cpu', dtype=torch.float32)
    cpp_fused_relu_view_6(c_void_p(buf28.data_ptr()), c_void_p(buf29.data_ptr()))
    buf30 = empty((128, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_9], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_17, buf29, reinterpret_tensor(primals_16, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf30)
    del primals_17
    buf31 = buf23; del buf23  # reuse
    buf32 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf34 = empty((1, 128, 1024), device='cpu', dtype=torch.float32)
    buf35 = empty((128, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_7(c_void_p(buf22.data_ptr()), c_void_p(buf30.data_ptr()), c_void_p(primals_18.data_ptr()), c_void_p(primals_19.data_ptr()), c_void_p(buf31.data_ptr()), c_void_p(buf32.data_ptr()), c_void_p(buf34.data_ptr()), c_void_p(buf35.data_ptr()))
    del primals_19
    buf36 = empty((128, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_encoder_layers_1_self_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_21, buf35, reinterpret_tensor(primals_20, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf36)
    del primals_21
    buf37 = empty((128, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_encoder_layers_1_self_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_23, buf35, reinterpret_tensor(primals_22, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf37)
    del primals_23
    buf38 = empty((128, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_encoder_layers_1_self_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_25, buf35, reinterpret_tensor(primals_24, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf38)
    del primals_25
    buf39 = empty((1, 16, 128, 64), device='cpu', dtype=torch.float32)
    buf40 = empty((1, 16, 128, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_8(c_void_p(buf36.data_ptr()), c_void_p(buf37.data_ptr()), c_void_p(buf39.data_ptr()), c_void_p(buf40.data_ptr()))
    buf41 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_weights_2], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf39, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf40, (16, 64, 128), (8192, 1, 64), 0), out=buf41)
    buf42 = empty((16, 128, 1), device='cpu', dtype=torch.float32)
    buf43 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    buf44 = empty((16, 128, 1), device='cpu', dtype=torch.float32)
    buf45 = buf43; del buf43  # reuse
    buf46 = reinterpret_tensor(buf37, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf37  # reuse
    cpp_fused__softmax_clone_9(c_void_p(buf45.data_ptr()), c_void_p(buf41.data_ptr()), c_void_p(buf38.data_ptr()), c_void_p(buf42.data_ptr()), c_void_p(buf44.data_ptr()), c_void_p(buf46.data_ptr()))
    buf47 = reinterpret_tensor(buf38, (16, 128, 64), (8192, 64, 1), 0); del buf38  # reuse
    # Source Nodes: [attn_output_5], Original ATen: [aten.bmm]
    extern_kernels.bmm(buf45, reinterpret_tensor(buf46, (16, 128, 64), (8192, 64, 1), 0), out=buf47)
    buf48 = buf36; del buf36  # reuse
    cpp_fused_view_10(c_void_p(buf47.data_ptr()), c_void_p(buf48.data_ptr()))
    buf49 = reinterpret_tensor(buf47, (128, 1024), (1024, 1), 0); del buf47  # reuse
    # Source Nodes: [hidden_states_14], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_27, buf48, reinterpret_tensor(primals_26, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf49)
    del primals_27
    buf50 = buf31; del buf31  # reuse
    buf51 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf53 = empty((1, 128, 1024), device='cpu', dtype=torch.float32)
    buf54 = empty((128, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_11(c_void_p(buf22.data_ptr()), c_void_p(buf30.data_ptr()), c_void_p(buf49.data_ptr()), c_void_p(primals_28.data_ptr()), c_void_p(primals_29.data_ptr()), c_void_p(buf50.data_ptr()), c_void_p(buf51.data_ptr()), c_void_p(buf53.data_ptr()), c_void_p(buf54.data_ptr()))
    del primals_29
    buf55 = empty((128, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_encoder_layers_1_fc1], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_31, buf54, reinterpret_tensor(primals_30, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf55)
    del primals_31
    buf56 = empty((128, 4096), device='cpu', dtype=torch.float32)
    cpp_fused_relu_view_12(c_void_p(buf55.data_ptr()), c_void_p(buf56.data_ptr()))
    buf57 = empty((128, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_20], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_33, buf56, reinterpret_tensor(primals_32, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf57)
    del primals_33
    buf58 = buf50; del buf50  # reuse
    buf59 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf61 = empty((1, 128, 1024), device='cpu', dtype=torch.float32)
    buf62 = empty((128, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_13(c_void_p(buf22.data_ptr()), c_void_p(buf30.data_ptr()), c_void_p(buf49.data_ptr()), c_void_p(buf57.data_ptr()), c_void_p(primals_34.data_ptr()), c_void_p(primals_35.data_ptr()), c_void_p(buf58.data_ptr()), c_void_p(buf59.data_ptr()), c_void_p(buf61.data_ptr()), c_void_p(buf62.data_ptr()))
    del primals_35
    buf63 = empty((128, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_encoder_layers_2_self_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_37, buf62, reinterpret_tensor(primals_36, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf63)
    del primals_37
    buf64 = empty((128, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_encoder_layers_2_self_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_39, buf62, reinterpret_tensor(primals_38, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf64)
    del primals_39
    buf65 = empty((128, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_encoder_layers_2_self_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_41, buf62, reinterpret_tensor(primals_40, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf65)
    del primals_41
    buf66 = empty((1, 16, 128, 64), device='cpu', dtype=torch.float32)
    buf67 = empty((1, 16, 128, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_14(c_void_p(buf63.data_ptr()), c_void_p(buf64.data_ptr()), c_void_p(buf66.data_ptr()), c_void_p(buf67.data_ptr()))
    buf68 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_weights_4], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf66, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf67, (16, 64, 128), (8192, 1, 64), 0), out=buf68)
    buf69 = empty((16, 128, 1), device='cpu', dtype=torch.float32)
    buf70 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    buf71 = empty((16, 128, 1), device='cpu', dtype=torch.float32)
    buf72 = buf70; del buf70  # reuse
    buf73 = reinterpret_tensor(buf64, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf64  # reuse
    cpp_fused__softmax_clone_15(c_void_p(buf72.data_ptr()), c_void_p(buf68.data_ptr()), c_void_p(buf65.data_ptr()), c_void_p(buf69.data_ptr()), c_void_p(buf71.data_ptr()), c_void_p(buf73.data_ptr()))
    buf74 = reinterpret_tensor(buf65, (16, 128, 64), (8192, 64, 1), 0); del buf65  # reuse
    # Source Nodes: [attn_output_10], Original ATen: [aten.bmm]
    extern_kernels.bmm(buf72, reinterpret_tensor(buf73, (16, 128, 64), (8192, 64, 1), 0), out=buf74)
    buf75 = buf63; del buf63  # reuse
    cpp_fused_view_16(c_void_p(buf74.data_ptr()), c_void_p(buf75.data_ptr()))
    buf76 = reinterpret_tensor(buf74, (128, 1024), (1024, 1), 0); del buf74  # reuse
    # Source Nodes: [hidden_states_25], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_43, buf75, reinterpret_tensor(primals_42, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf76)
    del primals_43
    buf77 = reinterpret_tensor(buf76, (1, 128, 1024), (131072, 1024, 1), 0); del buf76  # reuse
    buf78 = buf58; del buf58  # reuse
    buf79 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf81 = empty((1, 128, 1024), device='cpu', dtype=torch.float32)
    buf82 = empty((128, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_17(c_void_p(buf77.data_ptr()), c_void_p(buf22.data_ptr()), c_void_p(buf30.data_ptr()), c_void_p(buf49.data_ptr()), c_void_p(buf57.data_ptr()), c_void_p(primals_44.data_ptr()), c_void_p(primals_45.data_ptr()), c_void_p(buf78.data_ptr()), c_void_p(buf79.data_ptr()), c_void_p(buf81.data_ptr()), c_void_p(buf82.data_ptr()))
    del primals_45
    buf83 = empty((128, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_encoder_layers_2_fc1], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_47, buf82, reinterpret_tensor(primals_46, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf83)
    del primals_47
    buf84 = empty((128, 4096), device='cpu', dtype=torch.float32)
    cpp_fused_relu_view_18(c_void_p(buf83.data_ptr()), c_void_p(buf84.data_ptr()))
    buf85 = buf57; del buf57  # reuse
    # Source Nodes: [hidden_states_31], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_49, buf84, reinterpret_tensor(primals_48, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf85)
    del primals_49
    buf86 = buf78; del buf78  # reuse
    buf87 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf89 = reinterpret_tensor(buf49, (1, 128, 1024), (131072, 1024, 1), 0); del buf49  # reuse
    buf90 = buf30; del buf30  # reuse
    cpp_fused_add_native_layer_norm_view_19(c_void_p(buf77.data_ptr()), c_void_p(buf85.data_ptr()), c_void_p(primals_50.data_ptr()), c_void_p(primals_51.data_ptr()), c_void_p(buf86.data_ptr()), c_void_p(buf87.data_ptr()), c_void_p(buf89.data_ptr()), c_void_p(buf90.data_ptr()))
    del primals_51
    buf91 = reinterpret_tensor(buf22, (128, 1024), (1024, 1), 0); del buf22  # reuse
    # Source Nodes: [l__mod___model_encoder_layers_3_self_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_53, buf90, reinterpret_tensor(primals_52, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf91)
    del primals_53
    buf92 = empty((128, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_encoder_layers_3_self_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_55, buf90, reinterpret_tensor(primals_54, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf92)
    del primals_55
    buf93 = empty((128, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_encoder_layers_3_self_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_57, buf90, reinterpret_tensor(primals_56, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf93)
    del primals_57
    buf94 = empty((1, 16, 128, 64), device='cpu', dtype=torch.float32)
    buf95 = empty((1, 16, 128, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_20(c_void_p(buf91.data_ptr()), c_void_p(buf92.data_ptr()), c_void_p(buf94.data_ptr()), c_void_p(buf95.data_ptr()))
    buf96 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_weights_6], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf94, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf95, (16, 64, 128), (8192, 1, 64), 0), out=buf96)
    buf97 = empty((16, 128, 1), device='cpu', dtype=torch.float32)
    buf98 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    buf99 = empty((16, 128, 1), device='cpu', dtype=torch.float32)
    buf100 = buf98; del buf98  # reuse
    buf101 = reinterpret_tensor(buf92, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf92  # reuse
    cpp_fused__softmax_clone_21(c_void_p(buf100.data_ptr()), c_void_p(buf96.data_ptr()), c_void_p(buf93.data_ptr()), c_void_p(buf97.data_ptr()), c_void_p(buf99.data_ptr()), c_void_p(buf101.data_ptr()))
    buf102 = reinterpret_tensor(buf93, (16, 128, 64), (8192, 64, 1), 0); del buf93  # reuse
    # Source Nodes: [attn_output_15], Original ATen: [aten.bmm]
    extern_kernels.bmm(buf100, reinterpret_tensor(buf101, (16, 128, 64), (8192, 64, 1), 0), out=buf102)
    buf103 = buf91; del buf91  # reuse
    cpp_fused_view_22(c_void_p(buf102.data_ptr()), c_void_p(buf103.data_ptr()))
    buf104 = reinterpret_tensor(buf102, (128, 1024), (1024, 1), 0); del buf102  # reuse
    # Source Nodes: [hidden_states_36], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_59, buf103, reinterpret_tensor(primals_58, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf104)
    del primals_59
    buf105 = buf86; del buf86  # reuse
    buf106 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf108 = empty((1, 128, 1024), device='cpu', dtype=torch.float32)
    buf109 = empty((128, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_23(c_void_p(buf77.data_ptr()), c_void_p(buf85.data_ptr()), c_void_p(buf104.data_ptr()), c_void_p(primals_60.data_ptr()), c_void_p(primals_61.data_ptr()), c_void_p(buf105.data_ptr()), c_void_p(buf106.data_ptr()), c_void_p(buf108.data_ptr()), c_void_p(buf109.data_ptr()))
    del primals_61
    buf110 = empty((128, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_encoder_layers_3_fc1], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_63, buf109, reinterpret_tensor(primals_62, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf110)
    del primals_63
    buf111 = empty((128, 4096), device='cpu', dtype=torch.float32)
    cpp_fused_relu_view_24(c_void_p(buf110.data_ptr()), c_void_p(buf111.data_ptr()))
    buf112 = empty((128, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_42], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_65, buf111, reinterpret_tensor(primals_64, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf112)
    del primals_65
    buf113 = buf105; del buf105  # reuse
    buf114 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf116 = empty((1, 128, 1024), device='cpu', dtype=torch.float32)
    buf117 = empty((128, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_25(c_void_p(buf77.data_ptr()), c_void_p(buf85.data_ptr()), c_void_p(buf104.data_ptr()), c_void_p(buf112.data_ptr()), c_void_p(primals_66.data_ptr()), c_void_p(primals_67.data_ptr()), c_void_p(buf113.data_ptr()), c_void_p(buf114.data_ptr()), c_void_p(buf116.data_ptr()), c_void_p(buf117.data_ptr()))
    del primals_67
    buf118 = empty((128, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_encoder_layers_4_self_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_69, buf117, reinterpret_tensor(primals_68, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf118)
    del primals_69
    buf119 = empty((128, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_encoder_layers_4_self_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_71, buf117, reinterpret_tensor(primals_70, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf119)
    del primals_71
    buf120 = empty((128, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_encoder_layers_4_self_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_73, buf117, reinterpret_tensor(primals_72, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf120)
    del primals_73
    buf121 = empty((1, 16, 128, 64), device='cpu', dtype=torch.float32)
    buf122 = empty((1, 16, 128, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_26(c_void_p(buf118.data_ptr()), c_void_p(buf119.data_ptr()), c_void_p(buf121.data_ptr()), c_void_p(buf122.data_ptr()))
    buf123 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_weights_8], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf121, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf122, (16, 64, 128), (8192, 1, 64), 0), out=buf123)
    buf124 = empty((16, 128, 1), device='cpu', dtype=torch.float32)
    buf125 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    buf126 = empty((16, 128, 1), device='cpu', dtype=torch.float32)
    buf127 = buf125; del buf125  # reuse
    buf128 = reinterpret_tensor(buf119, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf119  # reuse
    cpp_fused__softmax_clone_27(c_void_p(buf127.data_ptr()), c_void_p(buf123.data_ptr()), c_void_p(buf120.data_ptr()), c_void_p(buf124.data_ptr()), c_void_p(buf126.data_ptr()), c_void_p(buf128.data_ptr()))
    buf129 = reinterpret_tensor(buf120, (16, 128, 64), (8192, 64, 1), 0); del buf120  # reuse
    # Source Nodes: [attn_output_20], Original ATen: [aten.bmm]
    extern_kernels.bmm(buf127, reinterpret_tensor(buf128, (16, 128, 64), (8192, 64, 1), 0), out=buf129)
    buf130 = buf118; del buf118  # reuse
    cpp_fused_view_28(c_void_p(buf129.data_ptr()), c_void_p(buf130.data_ptr()))
    buf131 = reinterpret_tensor(buf129, (128, 1024), (1024, 1), 0); del buf129  # reuse
    # Source Nodes: [hidden_states_47], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_75, buf130, reinterpret_tensor(primals_74, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf131)
    del primals_75
    buf132 = reinterpret_tensor(buf131, (1, 128, 1024), (131072, 1024, 1), 0); del buf131  # reuse
    buf133 = buf113; del buf113  # reuse
    buf134 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf136 = empty((1, 128, 1024), device='cpu', dtype=torch.float32)
    buf137 = empty((128, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_29(c_void_p(buf132.data_ptr()), c_void_p(buf77.data_ptr()), c_void_p(buf85.data_ptr()), c_void_p(buf104.data_ptr()), c_void_p(buf112.data_ptr()), c_void_p(primals_76.data_ptr()), c_void_p(primals_77.data_ptr()), c_void_p(buf133.data_ptr()), c_void_p(buf134.data_ptr()), c_void_p(buf136.data_ptr()), c_void_p(buf137.data_ptr()))
    del primals_77
    buf138 = empty((128, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_encoder_layers_4_fc1], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_79, buf137, reinterpret_tensor(primals_78, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf138)
    del primals_79
    buf139 = empty((128, 4096), device='cpu', dtype=torch.float32)
    cpp_fused_relu_view_30(c_void_p(buf138.data_ptr()), c_void_p(buf139.data_ptr()))
    buf140 = buf85; del buf85  # reuse
    # Source Nodes: [hidden_states_53], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_81, buf139, reinterpret_tensor(primals_80, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf140)
    del primals_81
    buf141 = buf133; del buf133  # reuse
    buf142 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf144 = buf77; del buf77  # reuse
    buf145 = buf112; del buf112  # reuse
    cpp_fused_add_native_layer_norm_view_31(c_void_p(buf132.data_ptr()), c_void_p(buf140.data_ptr()), c_void_p(primals_82.data_ptr()), c_void_p(primals_83.data_ptr()), c_void_p(buf141.data_ptr()), c_void_p(buf142.data_ptr()), c_void_p(buf144.data_ptr()), c_void_p(buf145.data_ptr()))
    del primals_83
    buf146 = buf104; del buf104  # reuse
    # Source Nodes: [l__mod___model_encoder_layers_5_self_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_85, buf145, reinterpret_tensor(primals_84, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf146)
    del primals_85
    buf147 = empty((128, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_encoder_layers_5_self_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_87, buf145, reinterpret_tensor(primals_86, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf147)
    del primals_87
    buf148 = empty((128, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_encoder_layers_5_self_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_89, buf145, reinterpret_tensor(primals_88, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf148)
    del primals_89
    buf149 = empty((1, 16, 128, 64), device='cpu', dtype=torch.float32)
    buf150 = empty((1, 16, 128, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_32(c_void_p(buf146.data_ptr()), c_void_p(buf147.data_ptr()), c_void_p(buf149.data_ptr()), c_void_p(buf150.data_ptr()))
    buf151 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_weights_10], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf149, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf150, (16, 64, 128), (8192, 1, 64), 0), out=buf151)
    buf152 = empty((16, 128, 1), device='cpu', dtype=torch.float32)
    buf153 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    buf154 = empty((16, 128, 1), device='cpu', dtype=torch.float32)
    buf155 = buf153; del buf153  # reuse
    buf156 = reinterpret_tensor(buf147, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf147  # reuse
    cpp_fused__softmax_clone_33(c_void_p(buf155.data_ptr()), c_void_p(buf151.data_ptr()), c_void_p(buf148.data_ptr()), c_void_p(buf152.data_ptr()), c_void_p(buf154.data_ptr()), c_void_p(buf156.data_ptr()))
    buf157 = reinterpret_tensor(buf148, (16, 128, 64), (8192, 64, 1), 0); del buf148  # reuse
    # Source Nodes: [attn_output_25], Original ATen: [aten.bmm]
    extern_kernels.bmm(buf155, reinterpret_tensor(buf156, (16, 128, 64), (8192, 64, 1), 0), out=buf157)
    buf158 = buf146; del buf146  # reuse
    cpp_fused_view_34(c_void_p(buf157.data_ptr()), c_void_p(buf158.data_ptr()))
    buf159 = reinterpret_tensor(buf157, (128, 1024), (1024, 1), 0); del buf157  # reuse
    # Source Nodes: [hidden_states_58], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_91, buf158, reinterpret_tensor(primals_90, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf159)
    del primals_91
    buf160 = buf141; del buf141  # reuse
    buf161 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf163 = empty((1, 128, 1024), device='cpu', dtype=torch.float32)
    buf164 = empty((128, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_35(c_void_p(buf132.data_ptr()), c_void_p(buf140.data_ptr()), c_void_p(buf159.data_ptr()), c_void_p(primals_92.data_ptr()), c_void_p(primals_93.data_ptr()), c_void_p(buf160.data_ptr()), c_void_p(buf161.data_ptr()), c_void_p(buf163.data_ptr()), c_void_p(buf164.data_ptr()))
    del primals_93
    buf165 = empty((128, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_encoder_layers_5_fc1], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_95, buf164, reinterpret_tensor(primals_94, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf165)
    del primals_95
    buf166 = empty((128, 4096), device='cpu', dtype=torch.float32)
    cpp_fused_relu_view_36(c_void_p(buf165.data_ptr()), c_void_p(buf166.data_ptr()))
    buf167 = empty((128, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_64], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_97, buf166, reinterpret_tensor(primals_96, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf167)
    del primals_97
    buf168 = buf160; del buf160  # reuse
    buf169 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf171 = empty((1, 128, 1024), device='cpu', dtype=torch.float32)
    buf172 = empty((128, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_37(c_void_p(buf132.data_ptr()), c_void_p(buf140.data_ptr()), c_void_p(buf159.data_ptr()), c_void_p(buf167.data_ptr()), c_void_p(primals_98.data_ptr()), c_void_p(primals_99.data_ptr()), c_void_p(buf168.data_ptr()), c_void_p(buf169.data_ptr()), c_void_p(buf171.data_ptr()), c_void_p(buf172.data_ptr()))
    del primals_99
    buf173 = empty((128, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_encoder_layers_6_self_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_101, buf172, reinterpret_tensor(primals_100, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf173)
    del primals_101
    buf174 = empty((128, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_encoder_layers_6_self_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_103, buf172, reinterpret_tensor(primals_102, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf174)
    del primals_103
    buf175 = empty((128, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_encoder_layers_6_self_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_105, buf172, reinterpret_tensor(primals_104, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf175)
    del primals_105
    buf176 = empty((1, 16, 128, 64), device='cpu', dtype=torch.float32)
    buf177 = empty((1, 16, 128, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_38(c_void_p(buf173.data_ptr()), c_void_p(buf174.data_ptr()), c_void_p(buf176.data_ptr()), c_void_p(buf177.data_ptr()))
    buf178 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_weights_12], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf176, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf177, (16, 64, 128), (8192, 1, 64), 0), out=buf178)
    buf179 = empty((16, 128, 1), device='cpu', dtype=torch.float32)
    buf180 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    buf181 = empty((16, 128, 1), device='cpu', dtype=torch.float32)
    buf182 = buf180; del buf180  # reuse
    buf183 = reinterpret_tensor(buf174, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf174  # reuse
    cpp_fused__softmax_clone_39(c_void_p(buf182.data_ptr()), c_void_p(buf178.data_ptr()), c_void_p(buf175.data_ptr()), c_void_p(buf179.data_ptr()), c_void_p(buf181.data_ptr()), c_void_p(buf183.data_ptr()))
    buf184 = reinterpret_tensor(buf175, (16, 128, 64), (8192, 64, 1), 0); del buf175  # reuse
    # Source Nodes: [attn_output_30], Original ATen: [aten.bmm]
    extern_kernels.bmm(buf182, reinterpret_tensor(buf183, (16, 128, 64), (8192, 64, 1), 0), out=buf184)
    buf185 = buf173; del buf173  # reuse
    cpp_fused_view_40(c_void_p(buf184.data_ptr()), c_void_p(buf185.data_ptr()))
    buf186 = reinterpret_tensor(buf184, (128, 1024), (1024, 1), 0); del buf184  # reuse
    # Source Nodes: [hidden_states_69], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_107, buf185, reinterpret_tensor(primals_106, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf186)
    del primals_107
    buf187 = reinterpret_tensor(buf186, (1, 128, 1024), (131072, 1024, 1), 0); del buf186  # reuse
    buf188 = buf168; del buf168  # reuse
    buf189 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf191 = empty((1, 128, 1024), device='cpu', dtype=torch.float32)
    buf192 = empty((128, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_41(c_void_p(buf187.data_ptr()), c_void_p(buf132.data_ptr()), c_void_p(buf140.data_ptr()), c_void_p(buf159.data_ptr()), c_void_p(buf167.data_ptr()), c_void_p(primals_108.data_ptr()), c_void_p(primals_109.data_ptr()), c_void_p(buf188.data_ptr()), c_void_p(buf189.data_ptr()), c_void_p(buf191.data_ptr()), c_void_p(buf192.data_ptr()))
    del primals_109
    buf193 = empty((128, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_encoder_layers_6_fc1], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_111, buf192, reinterpret_tensor(primals_110, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf193)
    del primals_111
    buf194 = empty((128, 4096), device='cpu', dtype=torch.float32)
    cpp_fused_relu_view_42(c_void_p(buf193.data_ptr()), c_void_p(buf194.data_ptr()))
    buf195 = buf167; del buf167  # reuse
    # Source Nodes: [hidden_states_75], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_113, buf194, reinterpret_tensor(primals_112, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf195)
    del primals_113
    buf196 = buf188; del buf188  # reuse
    buf197 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf199 = reinterpret_tensor(buf159, (1, 128, 1024), (131072, 1024, 1), 0); del buf159  # reuse
    buf200 = buf140; del buf140  # reuse
    cpp_fused_add_native_layer_norm_view_43(c_void_p(buf187.data_ptr()), c_void_p(buf195.data_ptr()), c_void_p(primals_114.data_ptr()), c_void_p(primals_115.data_ptr()), c_void_p(buf196.data_ptr()), c_void_p(buf197.data_ptr()), c_void_p(buf199.data_ptr()), c_void_p(buf200.data_ptr()))
    del primals_115
    buf201 = reinterpret_tensor(buf132, (128, 1024), (1024, 1), 0); del buf132  # reuse
    # Source Nodes: [l__mod___model_encoder_layers_7_self_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_117, buf200, reinterpret_tensor(primals_116, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf201)
    del primals_117
    buf202 = empty((128, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_encoder_layers_7_self_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_119, buf200, reinterpret_tensor(primals_118, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf202)
    del primals_119
    buf203 = empty((128, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_encoder_layers_7_self_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_121, buf200, reinterpret_tensor(primals_120, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf203)
    del primals_121
    buf204 = empty((1, 16, 128, 64), device='cpu', dtype=torch.float32)
    buf205 = empty((1, 16, 128, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_44(c_void_p(buf201.data_ptr()), c_void_p(buf202.data_ptr()), c_void_p(buf204.data_ptr()), c_void_p(buf205.data_ptr()))
    buf206 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_weights_14], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf204, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf205, (16, 64, 128), (8192, 1, 64), 0), out=buf206)
    buf207 = empty((16, 128, 1), device='cpu', dtype=torch.float32)
    buf208 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    buf209 = empty((16, 128, 1), device='cpu', dtype=torch.float32)
    buf210 = buf208; del buf208  # reuse
    buf211 = reinterpret_tensor(buf202, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf202  # reuse
    cpp_fused__softmax_clone_45(c_void_p(buf210.data_ptr()), c_void_p(buf206.data_ptr()), c_void_p(buf203.data_ptr()), c_void_p(buf207.data_ptr()), c_void_p(buf209.data_ptr()), c_void_p(buf211.data_ptr()))
    buf212 = reinterpret_tensor(buf203, (16, 128, 64), (8192, 64, 1), 0); del buf203  # reuse
    # Source Nodes: [attn_output_35], Original ATen: [aten.bmm]
    extern_kernels.bmm(buf210, reinterpret_tensor(buf211, (16, 128, 64), (8192, 64, 1), 0), out=buf212)
    buf213 = buf201; del buf201  # reuse
    cpp_fused_view_46(c_void_p(buf212.data_ptr()), c_void_p(buf213.data_ptr()))
    buf214 = reinterpret_tensor(buf212, (128, 1024), (1024, 1), 0); del buf212  # reuse
    # Source Nodes: [hidden_states_80], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_123, buf213, reinterpret_tensor(primals_122, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf214)
    del primals_123
    buf215 = buf196; del buf196  # reuse
    buf216 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf218 = empty((1, 128, 1024), device='cpu', dtype=torch.float32)
    buf219 = empty((128, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_47(c_void_p(buf187.data_ptr()), c_void_p(buf195.data_ptr()), c_void_p(buf214.data_ptr()), c_void_p(primals_124.data_ptr()), c_void_p(primals_125.data_ptr()), c_void_p(buf215.data_ptr()), c_void_p(buf216.data_ptr()), c_void_p(buf218.data_ptr()), c_void_p(buf219.data_ptr()))
    del primals_125
    buf220 = empty((128, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_encoder_layers_7_fc1], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_127, buf219, reinterpret_tensor(primals_126, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf220)
    del primals_127
    buf221 = empty((128, 4096), device='cpu', dtype=torch.float32)
    cpp_fused_relu_view_48(c_void_p(buf220.data_ptr()), c_void_p(buf221.data_ptr()))
    buf222 = empty((128, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_86], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_129, buf221, reinterpret_tensor(primals_128, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf222)
    del primals_129
    buf223 = buf215; del buf215  # reuse
    buf224 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf226 = empty((1, 128, 1024), device='cpu', dtype=torch.float32)
    buf227 = empty((128, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_49(c_void_p(buf187.data_ptr()), c_void_p(buf195.data_ptr()), c_void_p(buf214.data_ptr()), c_void_p(buf222.data_ptr()), c_void_p(primals_130.data_ptr()), c_void_p(primals_131.data_ptr()), c_void_p(buf223.data_ptr()), c_void_p(buf224.data_ptr()), c_void_p(buf226.data_ptr()), c_void_p(buf227.data_ptr()))
    del primals_131
    buf228 = empty((128, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_encoder_layers_8_self_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_133, buf227, reinterpret_tensor(primals_132, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf228)
    del primals_133
    buf229 = empty((128, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_encoder_layers_8_self_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_135, buf227, reinterpret_tensor(primals_134, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf229)
    del primals_135
    buf230 = empty((128, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_encoder_layers_8_self_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_137, buf227, reinterpret_tensor(primals_136, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf230)
    del primals_137
    buf231 = empty((1, 16, 128, 64), device='cpu', dtype=torch.float32)
    buf232 = empty((1, 16, 128, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_50(c_void_p(buf228.data_ptr()), c_void_p(buf229.data_ptr()), c_void_p(buf231.data_ptr()), c_void_p(buf232.data_ptr()))
    buf233 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_weights_16], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf231, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf232, (16, 64, 128), (8192, 1, 64), 0), out=buf233)
    buf234 = empty((16, 128, 1), device='cpu', dtype=torch.float32)
    buf235 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    buf236 = empty((16, 128, 1), device='cpu', dtype=torch.float32)
    buf237 = buf235; del buf235  # reuse
    buf238 = reinterpret_tensor(buf229, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf229  # reuse
    cpp_fused__softmax_clone_51(c_void_p(buf237.data_ptr()), c_void_p(buf233.data_ptr()), c_void_p(buf230.data_ptr()), c_void_p(buf234.data_ptr()), c_void_p(buf236.data_ptr()), c_void_p(buf238.data_ptr()))
    buf239 = reinterpret_tensor(buf230, (16, 128, 64), (8192, 64, 1), 0); del buf230  # reuse
    # Source Nodes: [attn_output_40], Original ATen: [aten.bmm]
    extern_kernels.bmm(buf237, reinterpret_tensor(buf238, (16, 128, 64), (8192, 64, 1), 0), out=buf239)
    buf240 = buf228; del buf228  # reuse
    cpp_fused_view_52(c_void_p(buf239.data_ptr()), c_void_p(buf240.data_ptr()))
    buf241 = reinterpret_tensor(buf239, (128, 1024), (1024, 1), 0); del buf239  # reuse
    # Source Nodes: [hidden_states_91], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_139, buf240, reinterpret_tensor(primals_138, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf241)
    del primals_139
    buf242 = reinterpret_tensor(buf241, (1, 128, 1024), (131072, 1024, 1), 0); del buf241  # reuse
    buf243 = buf223; del buf223  # reuse
    buf244 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf246 = empty((1, 128, 1024), device='cpu', dtype=torch.float32)
    buf247 = empty((128, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_53(c_void_p(buf242.data_ptr()), c_void_p(buf187.data_ptr()), c_void_p(buf195.data_ptr()), c_void_p(buf214.data_ptr()), c_void_p(buf222.data_ptr()), c_void_p(primals_140.data_ptr()), c_void_p(primals_141.data_ptr()), c_void_p(buf243.data_ptr()), c_void_p(buf244.data_ptr()), c_void_p(buf246.data_ptr()), c_void_p(buf247.data_ptr()))
    del primals_141
    buf248 = empty((128, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_encoder_layers_8_fc1], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_143, buf247, reinterpret_tensor(primals_142, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf248)
    del primals_143
    buf249 = empty((128, 4096), device='cpu', dtype=torch.float32)
    cpp_fused_relu_view_54(c_void_p(buf248.data_ptr()), c_void_p(buf249.data_ptr()))
    buf250 = buf222; del buf222  # reuse
    # Source Nodes: [hidden_states_97], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_145, buf249, reinterpret_tensor(primals_144, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf250)
    del primals_145
    buf251 = buf243; del buf243  # reuse
    buf252 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf254 = reinterpret_tensor(buf214, (1, 128, 1024), (131072, 1024, 1), 0); del buf214  # reuse
    buf255 = buf195; del buf195  # reuse
    cpp_fused_add_native_layer_norm_view_55(c_void_p(buf242.data_ptr()), c_void_p(buf250.data_ptr()), c_void_p(primals_146.data_ptr()), c_void_p(primals_147.data_ptr()), c_void_p(buf251.data_ptr()), c_void_p(buf252.data_ptr()), c_void_p(buf254.data_ptr()), c_void_p(buf255.data_ptr()))
    del primals_147
    buf256 = reinterpret_tensor(buf187, (128, 1024), (1024, 1), 0); del buf187  # reuse
    # Source Nodes: [l__mod___model_encoder_layers_9_self_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_149, buf255, reinterpret_tensor(primals_148, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf256)
    del primals_149
    buf257 = empty((128, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_encoder_layers_9_self_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_151, buf255, reinterpret_tensor(primals_150, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf257)
    del primals_151
    buf258 = empty((128, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_encoder_layers_9_self_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_153, buf255, reinterpret_tensor(primals_152, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf258)
    del primals_153
    buf259 = empty((1, 16, 128, 64), device='cpu', dtype=torch.float32)
    buf260 = empty((1, 16, 128, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_56(c_void_p(buf256.data_ptr()), c_void_p(buf257.data_ptr()), c_void_p(buf259.data_ptr()), c_void_p(buf260.data_ptr()))
    buf261 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_weights_18], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf259, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf260, (16, 64, 128), (8192, 1, 64), 0), out=buf261)
    buf262 = empty((16, 128, 1), device='cpu', dtype=torch.float32)
    buf263 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    buf264 = empty((16, 128, 1), device='cpu', dtype=torch.float32)
    buf265 = buf263; del buf263  # reuse
    buf266 = reinterpret_tensor(buf257, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf257  # reuse
    cpp_fused__softmax_clone_57(c_void_p(buf265.data_ptr()), c_void_p(buf261.data_ptr()), c_void_p(buf258.data_ptr()), c_void_p(buf262.data_ptr()), c_void_p(buf264.data_ptr()), c_void_p(buf266.data_ptr()))
    buf267 = reinterpret_tensor(buf258, (16, 128, 64), (8192, 64, 1), 0); del buf258  # reuse
    # Source Nodes: [attn_output_45], Original ATen: [aten.bmm]
    extern_kernels.bmm(buf265, reinterpret_tensor(buf266, (16, 128, 64), (8192, 64, 1), 0), out=buf267)
    buf268 = buf256; del buf256  # reuse
    cpp_fused_view_58(c_void_p(buf267.data_ptr()), c_void_p(buf268.data_ptr()))
    buf269 = reinterpret_tensor(buf267, (128, 1024), (1024, 1), 0); del buf267  # reuse
    # Source Nodes: [hidden_states_102], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_155, buf268, reinterpret_tensor(primals_154, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf269)
    del primals_155
    buf270 = buf251; del buf251  # reuse
    buf271 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf273 = empty((1, 128, 1024), device='cpu', dtype=torch.float32)
    buf274 = empty((128, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_59(c_void_p(buf242.data_ptr()), c_void_p(buf250.data_ptr()), c_void_p(buf269.data_ptr()), c_void_p(primals_156.data_ptr()), c_void_p(primals_157.data_ptr()), c_void_p(buf270.data_ptr()), c_void_p(buf271.data_ptr()), c_void_p(buf273.data_ptr()), c_void_p(buf274.data_ptr()))
    del primals_157
    buf275 = empty((128, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_encoder_layers_9_fc1], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_159, buf274, reinterpret_tensor(primals_158, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf275)
    del primals_159
    buf276 = empty((128, 4096), device='cpu', dtype=torch.float32)
    cpp_fused_relu_view_60(c_void_p(buf275.data_ptr()), c_void_p(buf276.data_ptr()))
    buf277 = empty((128, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_108], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_161, buf276, reinterpret_tensor(primals_160, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf277)
    del primals_161
    buf278 = buf270; del buf270  # reuse
    buf279 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf281 = empty((1, 128, 1024), device='cpu', dtype=torch.float32)
    buf282 = empty((128, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_61(c_void_p(buf242.data_ptr()), c_void_p(buf250.data_ptr()), c_void_p(buf269.data_ptr()), c_void_p(buf277.data_ptr()), c_void_p(primals_162.data_ptr()), c_void_p(primals_163.data_ptr()), c_void_p(buf278.data_ptr()), c_void_p(buf279.data_ptr()), c_void_p(buf281.data_ptr()), c_void_p(buf282.data_ptr()))
    del primals_163
    buf283 = empty((128, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_encoder_layers_10_self_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_165, buf282, reinterpret_tensor(primals_164, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf283)
    del primals_165
    buf284 = empty((128, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_encoder_layers_10_self_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_167, buf282, reinterpret_tensor(primals_166, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf284)
    del primals_167
    buf285 = empty((128, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_encoder_layers_10_self_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_169, buf282, reinterpret_tensor(primals_168, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf285)
    del primals_169
    buf286 = empty((1, 16, 128, 64), device='cpu', dtype=torch.float32)
    buf287 = empty((1, 16, 128, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_62(c_void_p(buf283.data_ptr()), c_void_p(buf284.data_ptr()), c_void_p(buf286.data_ptr()), c_void_p(buf287.data_ptr()))
    buf288 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_weights_20], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf286, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf287, (16, 64, 128), (8192, 1, 64), 0), out=buf288)
    buf289 = empty((16, 128, 1), device='cpu', dtype=torch.float32)
    buf290 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    buf291 = empty((16, 128, 1), device='cpu', dtype=torch.float32)
    buf292 = buf290; del buf290  # reuse
    buf293 = reinterpret_tensor(buf284, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf284  # reuse
    cpp_fused__softmax_clone_63(c_void_p(buf292.data_ptr()), c_void_p(buf288.data_ptr()), c_void_p(buf285.data_ptr()), c_void_p(buf289.data_ptr()), c_void_p(buf291.data_ptr()), c_void_p(buf293.data_ptr()))
    buf294 = reinterpret_tensor(buf285, (16, 128, 64), (8192, 64, 1), 0); del buf285  # reuse
    # Source Nodes: [attn_output_50], Original ATen: [aten.bmm]
    extern_kernels.bmm(buf292, reinterpret_tensor(buf293, (16, 128, 64), (8192, 64, 1), 0), out=buf294)
    buf295 = buf283; del buf283  # reuse
    cpp_fused_view_64(c_void_p(buf294.data_ptr()), c_void_p(buf295.data_ptr()))
    buf296 = reinterpret_tensor(buf294, (128, 1024), (1024, 1), 0); del buf294  # reuse
    # Source Nodes: [hidden_states_113], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_171, buf295, reinterpret_tensor(primals_170, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf296)
    del primals_171
    buf297 = reinterpret_tensor(buf296, (1, 128, 1024), (131072, 1024, 1), 0); del buf296  # reuse
    buf298 = buf278; del buf278  # reuse
    buf299 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf301 = empty((1, 128, 1024), device='cpu', dtype=torch.float32)
    buf302 = empty((128, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_65(c_void_p(buf297.data_ptr()), c_void_p(buf242.data_ptr()), c_void_p(buf250.data_ptr()), c_void_p(buf269.data_ptr()), c_void_p(buf277.data_ptr()), c_void_p(primals_172.data_ptr()), c_void_p(primals_173.data_ptr()), c_void_p(buf298.data_ptr()), c_void_p(buf299.data_ptr()), c_void_p(buf301.data_ptr()), c_void_p(buf302.data_ptr()))
    del primals_173
    buf303 = empty((128, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_encoder_layers_10_fc1], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_175, buf302, reinterpret_tensor(primals_174, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf303)
    del primals_175
    buf304 = empty((128, 4096), device='cpu', dtype=torch.float32)
    cpp_fused_relu_view_66(c_void_p(buf303.data_ptr()), c_void_p(buf304.data_ptr()))
    buf305 = buf277; del buf277  # reuse
    # Source Nodes: [hidden_states_119], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_177, buf304, reinterpret_tensor(primals_176, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf305)
    del primals_177
    buf306 = buf298; del buf298  # reuse
    buf307 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf309 = reinterpret_tensor(buf269, (1, 128, 1024), (131072, 1024, 1), 0); del buf269  # reuse
    buf310 = buf250; del buf250  # reuse
    cpp_fused_add_native_layer_norm_view_67(c_void_p(buf297.data_ptr()), c_void_p(buf305.data_ptr()), c_void_p(primals_178.data_ptr()), c_void_p(primals_179.data_ptr()), c_void_p(buf306.data_ptr()), c_void_p(buf307.data_ptr()), c_void_p(buf309.data_ptr()), c_void_p(buf310.data_ptr()))
    del primals_179
    buf311 = reinterpret_tensor(buf242, (128, 1024), (1024, 1), 0); del buf242  # reuse
    # Source Nodes: [l__mod___model_encoder_layers_11_self_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_181, buf310, reinterpret_tensor(primals_180, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf311)
    del primals_181
    buf312 = empty((128, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_encoder_layers_11_self_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_183, buf310, reinterpret_tensor(primals_182, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf312)
    del primals_183
    buf313 = empty((128, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_encoder_layers_11_self_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_185, buf310, reinterpret_tensor(primals_184, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf313)
    del primals_185
    buf314 = empty((1, 16, 128, 64), device='cpu', dtype=torch.float32)
    buf315 = empty((1, 16, 128, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_68(c_void_p(buf311.data_ptr()), c_void_p(buf312.data_ptr()), c_void_p(buf314.data_ptr()), c_void_p(buf315.data_ptr()))
    buf316 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_weights_22], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf314, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf315, (16, 64, 128), (8192, 1, 64), 0), out=buf316)
    buf317 = empty((16, 128, 1), device='cpu', dtype=torch.float32)
    buf318 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    buf319 = empty((16, 128, 1), device='cpu', dtype=torch.float32)
    buf320 = buf318; del buf318  # reuse
    buf321 = reinterpret_tensor(buf312, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf312  # reuse
    cpp_fused__softmax_clone_69(c_void_p(buf320.data_ptr()), c_void_p(buf316.data_ptr()), c_void_p(buf313.data_ptr()), c_void_p(buf317.data_ptr()), c_void_p(buf319.data_ptr()), c_void_p(buf321.data_ptr()))
    buf322 = reinterpret_tensor(buf313, (16, 128, 64), (8192, 64, 1), 0); del buf313  # reuse
    # Source Nodes: [attn_output_55], Original ATen: [aten.bmm]
    extern_kernels.bmm(buf320, reinterpret_tensor(buf321, (16, 128, 64), (8192, 64, 1), 0), out=buf322)
    buf323 = buf311; del buf311  # reuse
    cpp_fused_view_70(c_void_p(buf322.data_ptr()), c_void_p(buf323.data_ptr()))
    buf324 = reinterpret_tensor(buf322, (128, 1024), (1024, 1), 0); del buf322  # reuse
    # Source Nodes: [hidden_states_124], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_187, buf323, reinterpret_tensor(primals_186, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf324)
    del primals_187
    buf325 = buf306; del buf306  # reuse
    buf326 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf328 = empty((1, 128, 1024), device='cpu', dtype=torch.float32)
    buf329 = empty((128, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_71(c_void_p(buf297.data_ptr()), c_void_p(buf305.data_ptr()), c_void_p(buf324.data_ptr()), c_void_p(primals_188.data_ptr()), c_void_p(primals_189.data_ptr()), c_void_p(buf325.data_ptr()), c_void_p(buf326.data_ptr()), c_void_p(buf328.data_ptr()), c_void_p(buf329.data_ptr()))
    del primals_189
    buf330 = empty((128, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_encoder_layers_11_fc1], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_191, buf329, reinterpret_tensor(primals_190, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf330)
    del primals_191
    buf331 = empty((128, 4096), device='cpu', dtype=torch.float32)
    cpp_fused_relu_view_72(c_void_p(buf330.data_ptr()), c_void_p(buf331.data_ptr()))
    buf332 = empty((128, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_130], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_193, buf331, reinterpret_tensor(primals_192, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf332)
    del primals_193
    buf333 = buf325; del buf325  # reuse
    buf334 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf336 = empty((1, 128, 1024), device='cpu', dtype=torch.float32)
    buf337 = empty((1, 128, 1024), device='cpu', dtype=torch.float32)
    buf338 = buf0; del buf0  # reuse
    cpp_fused__to_copy_add_cumsum_native_layer_norm_ne_73(c_void_p(buf297.data_ptr()), c_void_p(buf305.data_ptr()), c_void_p(buf324.data_ptr()), c_void_p(buf332.data_ptr()), c_void_p(primals_194.data_ptr()), c_void_p(primals_195.data_ptr()), c_void_p(primals_515.data_ptr()), c_void_p(buf333.data_ptr()), c_void_p(buf334.data_ptr()), c_void_p(buf336.data_ptr()), c_void_p(buf337.data_ptr()), c_void_p(buf338.data_ptr()))
    del primals_195
    # Source Nodes: [cumsum_1, mask_3, ne_1], Original ATen: [aten._to_copy, aten.cumsum, aten.ne]
    buf339 = aten.cumsum(buf338, 1)
    del buf338
    buf340 = buf339
    del buf339
    buf341 = buf333; del buf333  # reuse
    buf342 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf344 = reinterpret_tensor(buf332, (1, 128, 1024), (131072, 1024, 1), 0); del buf332  # reuse
    buf345 = buf324; del buf324  # reuse
    cpp_fused_add_embedding_mul_native_layer_norm_view_74(c_void_p(primals_515.data_ptr()), c_void_p(primals_196.data_ptr()), c_void_p(buf340.data_ptr()), c_void_p(primals_513.data_ptr()), c_void_p(primals_197.data_ptr()), c_void_p(primals_198.data_ptr()), c_void_p(buf341.data_ptr()), c_void_p(buf342.data_ptr()), c_void_p(buf344.data_ptr()), c_void_p(buf345.data_ptr()))
    del primals_198
    buf346 = buf305; del buf305  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_0_self_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_200, buf345, reinterpret_tensor(primals_199, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf346)
    del primals_200
    buf347 = reinterpret_tensor(buf297, (128, 1024), (1024, 1), 0); del buf297  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_0_self_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_202, buf345, reinterpret_tensor(primals_201, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf347)
    del primals_202
    buf348 = empty((1, 16, 128, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_75(c_void_p(buf347.data_ptr()), c_void_p(buf348.data_ptr()))
    buf349 = buf347; del buf347  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_0_self_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_204, buf345, reinterpret_tensor(primals_203, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf349)
    del primals_204
    buf350 = empty((1, 16, 128, 64), device='cpu', dtype=torch.float32)
    buf351 = empty((1, 16, 128, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_76(c_void_p(buf349.data_ptr()), c_void_p(buf346.data_ptr()), c_void_p(buf350.data_ptr()), c_void_p(buf351.data_ptr()))
    buf352 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_weights_24], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf351, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf348, (16, 64, 128), (8192, 1, 64), 0), out=buf352)
    buf353 = empty_strided((16, 128, 1), (128, 1, 2048), device='cpu', dtype=torch.float32)
    buf354 = buf352; del buf352  # reuse
    buf355 = empty_strided((16, 128, 1), (128, 1, 2048), device='cpu', dtype=torch.float32)
    buf356 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_77(c_void_p(buf354.data_ptr()), c_void_p(buf353.data_ptr()), c_void_p(buf355.data_ptr()), c_void_p(buf356.data_ptr()))
    buf357 = reinterpret_tensor(buf349, (16, 128, 64), (8192, 64, 1), 0); del buf349  # reuse
    # Source Nodes: [attn_output_60], Original ATen: [aten.bmm]
    extern_kernels.bmm(buf356, reinterpret_tensor(buf350, (16, 128, 64), (8192, 64, 1), 0), out=buf357)
    buf358 = buf346; del buf346  # reuse
    cpp_fused_view_78(c_void_p(buf357.data_ptr()), c_void_p(buf358.data_ptr()))
    buf359 = reinterpret_tensor(buf357, (128, 1024), (1024, 1), 0); del buf357  # reuse
    # Source Nodes: [hidden_states_138], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_206, buf358, reinterpret_tensor(primals_205, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf359)
    del primals_206
    buf360 = reinterpret_tensor(buf359, (1, 128, 1024), (131072, 1024, 1), 0); del buf359  # reuse
    buf361 = buf341; del buf341  # reuse
    buf362 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf364 = empty((1, 128, 1024), device='cpu', dtype=torch.float32)
    buf365 = empty((128, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_embedding_mul_native_layer_norm_view_79(c_void_p(buf360.data_ptr()), c_void_p(primals_515.data_ptr()), c_void_p(primals_196.data_ptr()), c_void_p(buf340.data_ptr()), c_void_p(primals_513.data_ptr()), c_void_p(primals_207.data_ptr()), c_void_p(primals_208.data_ptr()), c_void_p(buf361.data_ptr()), c_void_p(buf362.data_ptr()), c_void_p(buf364.data_ptr()), c_void_p(buf365.data_ptr()))
    del buf340
    del primals_196
    del primals_208
    del primals_513
    buf366 = empty((128, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_decoder_layers_0_encoder_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_210, buf365, reinterpret_tensor(primals_209, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf366)
    del primals_210
    buf367 = empty((128, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_decoder_layers_0_encoder_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_212, reinterpret_tensor(buf337, (128, 1024), (1024, 1), 0), reinterpret_tensor(primals_211, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf367)
    del primals_212
    buf368 = empty((1, 16, 128, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_80(c_void_p(buf367.data_ptr()), c_void_p(buf368.data_ptr()))
    buf369 = buf367; del buf367  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_0_encoder_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_214, reinterpret_tensor(buf337, (128, 1024), (1024, 1), 0), reinterpret_tensor(primals_213, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf369)
    del primals_214
    buf370 = empty((1, 16, 128, 64), device='cpu', dtype=torch.float32)
    buf371 = empty((1, 16, 128, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_81(c_void_p(buf369.data_ptr()), c_void_p(buf366.data_ptr()), c_void_p(buf370.data_ptr()), c_void_p(buf371.data_ptr()))
    buf372 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_weights_28], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf371, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf368, (16, 64, 128), (8192, 1, 64), 0), out=buf372)
    buf373 = reinterpret_tensor(buf353, (16, 128, 1), (128, 1, 1), 0); del buf353  # reuse
    buf374 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    buf375 = empty((16, 128, 1), device='cpu', dtype=torch.float32)
    buf376 = buf374; del buf374  # reuse
    cpp_fused__softmax_82(c_void_p(buf376.data_ptr()), c_void_p(buf372.data_ptr()), c_void_p(buf373.data_ptr()), c_void_p(buf375.data_ptr()))
    buf377 = reinterpret_tensor(buf369, (16, 128, 64), (8192, 64, 1), 0); del buf369  # reuse
    # Source Nodes: [attn_output_65], Original ATen: [aten.bmm]
    extern_kernels.bmm(buf376, reinterpret_tensor(buf370, (16, 128, 64), (8192, 64, 1), 0), out=buf377)
    buf378 = buf366; del buf366  # reuse
    cpp_fused_view_83(c_void_p(buf377.data_ptr()), c_void_p(buf378.data_ptr()))
    buf379 = reinterpret_tensor(buf377, (128, 1024), (1024, 1), 0); del buf377  # reuse
    # Source Nodes: [hidden_states_142], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_216, buf378, reinterpret_tensor(primals_215, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf379)
    del primals_216
    buf380 = buf361; del buf361  # reuse
    buf381 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf383 = empty((1, 128, 1024), device='cpu', dtype=torch.float32)
    buf384 = empty((128, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_84(c_void_p(buf360.data_ptr()), c_void_p(buf379.data_ptr()), c_void_p(primals_217.data_ptr()), c_void_p(primals_218.data_ptr()), c_void_p(buf380.data_ptr()), c_void_p(buf381.data_ptr()), c_void_p(buf383.data_ptr()), c_void_p(buf384.data_ptr()))
    del primals_218
    buf385 = empty((128, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_decoder_layers_0_fc1], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_220, buf384, reinterpret_tensor(primals_219, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf385)
    del primals_220
    buf386 = empty((128, 4096), device='cpu', dtype=torch.float32)
    cpp_fused_relu_view_85(c_void_p(buf385.data_ptr()), c_void_p(buf386.data_ptr()))
    buf387 = empty((128, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_148], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_222, buf386, reinterpret_tensor(primals_221, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf387)
    del primals_222
    buf388 = buf380; del buf380  # reuse
    buf389 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf391 = empty((1, 128, 1024), device='cpu', dtype=torch.float32)
    buf392 = empty((128, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_86(c_void_p(buf360.data_ptr()), c_void_p(buf379.data_ptr()), c_void_p(buf387.data_ptr()), c_void_p(primals_223.data_ptr()), c_void_p(primals_224.data_ptr()), c_void_p(buf388.data_ptr()), c_void_p(buf389.data_ptr()), c_void_p(buf391.data_ptr()), c_void_p(buf392.data_ptr()))
    del primals_224
    buf393 = empty((128, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_decoder_layers_1_self_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_226, buf392, reinterpret_tensor(primals_225, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf393)
    del primals_226
    buf394 = empty((128, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_decoder_layers_1_self_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_228, buf392, reinterpret_tensor(primals_227, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf394)
    del primals_228
    buf395 = empty((1, 16, 128, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_87(c_void_p(buf394.data_ptr()), c_void_p(buf395.data_ptr()))
    buf396 = buf394; del buf394  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_1_self_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_230, buf392, reinterpret_tensor(primals_229, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf396)
    del primals_230
    buf397 = empty((1, 16, 128, 64), device='cpu', dtype=torch.float32)
    buf398 = empty((1, 16, 128, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_88(c_void_p(buf396.data_ptr()), c_void_p(buf393.data_ptr()), c_void_p(buf397.data_ptr()), c_void_p(buf398.data_ptr()))
    buf399 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_weights_30], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf398, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf395, (16, 64, 128), (8192, 1, 64), 0), out=buf399)
    buf400 = empty_strided((16, 128, 1), (128, 1, 2048), device='cpu', dtype=torch.float32)
    buf401 = buf399; del buf399  # reuse
    buf402 = empty_strided((16, 128, 1), (128, 1, 2048), device='cpu', dtype=torch.float32)
    buf403 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_89(c_void_p(buf401.data_ptr()), c_void_p(buf400.data_ptr()), c_void_p(buf402.data_ptr()), c_void_p(buf403.data_ptr()))
    buf404 = reinterpret_tensor(buf396, (16, 128, 64), (8192, 64, 1), 0); del buf396  # reuse
    # Source Nodes: [attn_output_70], Original ATen: [aten.bmm]
    extern_kernels.bmm(buf403, reinterpret_tensor(buf397, (16, 128, 64), (8192, 64, 1), 0), out=buf404)
    buf405 = buf393; del buf393  # reuse
    cpp_fused_view_90(c_void_p(buf404.data_ptr()), c_void_p(buf405.data_ptr()))
    buf406 = reinterpret_tensor(buf404, (128, 1024), (1024, 1), 0); del buf404  # reuse
    # Source Nodes: [hidden_states_153], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_232, buf405, reinterpret_tensor(primals_231, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf406)
    del primals_232
    buf407 = buf388; del buf388  # reuse
    buf408 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf410 = empty((1, 128, 1024), device='cpu', dtype=torch.float32)
    buf411 = empty((128, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_91(c_void_p(buf360.data_ptr()), c_void_p(buf379.data_ptr()), c_void_p(buf387.data_ptr()), c_void_p(buf406.data_ptr()), c_void_p(primals_233.data_ptr()), c_void_p(primals_234.data_ptr()), c_void_p(buf407.data_ptr()), c_void_p(buf408.data_ptr()), c_void_p(buf410.data_ptr()), c_void_p(buf411.data_ptr()))
    del primals_234
    buf412 = empty((128, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_decoder_layers_1_encoder_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_236, buf411, reinterpret_tensor(primals_235, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf412)
    del primals_236
    buf413 = empty((128, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_decoder_layers_1_encoder_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_238, reinterpret_tensor(buf337, (128, 1024), (1024, 1), 0), reinterpret_tensor(primals_237, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf413)
    del primals_238
    buf414 = empty((1, 16, 128, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_92(c_void_p(buf413.data_ptr()), c_void_p(buf414.data_ptr()))
    buf415 = buf413; del buf413  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_1_encoder_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_240, reinterpret_tensor(buf337, (128, 1024), (1024, 1), 0), reinterpret_tensor(primals_239, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf415)
    del primals_240
    buf416 = empty((1, 16, 128, 64), device='cpu', dtype=torch.float32)
    buf417 = empty((1, 16, 128, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_93(c_void_p(buf415.data_ptr()), c_void_p(buf412.data_ptr()), c_void_p(buf416.data_ptr()), c_void_p(buf417.data_ptr()))
    buf418 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_weights_34], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf417, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf414, (16, 64, 128), (8192, 1, 64), 0), out=buf418)
    buf419 = reinterpret_tensor(buf400, (16, 128, 1), (128, 1, 1), 0); del buf400  # reuse
    buf420 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    buf421 = empty((16, 128, 1), device='cpu', dtype=torch.float32)
    buf422 = buf420; del buf420  # reuse
    cpp_fused__softmax_94(c_void_p(buf422.data_ptr()), c_void_p(buf418.data_ptr()), c_void_p(buf419.data_ptr()), c_void_p(buf421.data_ptr()))
    buf423 = reinterpret_tensor(buf415, (16, 128, 64), (8192, 64, 1), 0); del buf415  # reuse
    # Source Nodes: [attn_output_75], Original ATen: [aten.bmm]
    extern_kernels.bmm(buf422, reinterpret_tensor(buf416, (16, 128, 64), (8192, 64, 1), 0), out=buf423)
    buf424 = buf412; del buf412  # reuse
    cpp_fused_view_95(c_void_p(buf423.data_ptr()), c_void_p(buf424.data_ptr()))
    buf425 = reinterpret_tensor(buf423, (128, 1024), (1024, 1), 0); del buf423  # reuse
    # Source Nodes: [hidden_states_157], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_242, buf424, reinterpret_tensor(primals_241, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf425)
    del primals_242
    buf426 = reinterpret_tensor(buf425, (1, 128, 1024), (131072, 1024, 1), 0); del buf425  # reuse
    buf427 = buf407; del buf407  # reuse
    buf428 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf430 = empty((1, 128, 1024), device='cpu', dtype=torch.float32)
    buf431 = empty((128, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_96(c_void_p(buf426.data_ptr()), c_void_p(buf360.data_ptr()), c_void_p(buf379.data_ptr()), c_void_p(buf387.data_ptr()), c_void_p(buf406.data_ptr()), c_void_p(primals_243.data_ptr()), c_void_p(primals_244.data_ptr()), c_void_p(buf427.data_ptr()), c_void_p(buf428.data_ptr()), c_void_p(buf430.data_ptr()), c_void_p(buf431.data_ptr()))
    del primals_244
    buf432 = empty((128, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_decoder_layers_1_fc1], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_246, buf431, reinterpret_tensor(primals_245, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf432)
    del primals_246
    buf433 = empty((128, 4096), device='cpu', dtype=torch.float32)
    cpp_fused_relu_view_97(c_void_p(buf432.data_ptr()), c_void_p(buf433.data_ptr()))
    buf434 = buf406; del buf406  # reuse
    # Source Nodes: [hidden_states_163], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_248, buf433, reinterpret_tensor(primals_247, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf434)
    del primals_248
    buf435 = buf427; del buf427  # reuse
    buf436 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf438 = reinterpret_tensor(buf387, (1, 128, 1024), (131072, 1024, 1), 0); del buf387  # reuse
    buf439 = buf379; del buf379  # reuse
    cpp_fused_add_native_layer_norm_view_98(c_void_p(buf426.data_ptr()), c_void_p(buf434.data_ptr()), c_void_p(primals_249.data_ptr()), c_void_p(primals_250.data_ptr()), c_void_p(buf435.data_ptr()), c_void_p(buf436.data_ptr()), c_void_p(buf438.data_ptr()), c_void_p(buf439.data_ptr()))
    del primals_250
    buf440 = reinterpret_tensor(buf360, (128, 1024), (1024, 1), 0); del buf360  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_2_self_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_252, buf439, reinterpret_tensor(primals_251, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf440)
    del primals_252
    buf441 = empty((128, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_decoder_layers_2_self_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_254, buf439, reinterpret_tensor(primals_253, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf441)
    del primals_254
    buf442 = empty((1, 16, 128, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_99(c_void_p(buf441.data_ptr()), c_void_p(buf442.data_ptr()))
    buf443 = buf441; del buf441  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_2_self_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_256, buf439, reinterpret_tensor(primals_255, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf443)
    del primals_256
    buf444 = empty((1, 16, 128, 64), device='cpu', dtype=torch.float32)
    buf445 = empty((1, 16, 128, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_100(c_void_p(buf443.data_ptr()), c_void_p(buf440.data_ptr()), c_void_p(buf444.data_ptr()), c_void_p(buf445.data_ptr()))
    buf446 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_weights_36], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf445, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf442, (16, 64, 128), (8192, 1, 64), 0), out=buf446)
    buf447 = empty_strided((16, 128, 1), (128, 1, 2048), device='cpu', dtype=torch.float32)
    buf448 = buf446; del buf446  # reuse
    buf449 = empty_strided((16, 128, 1), (128, 1, 2048), device='cpu', dtype=torch.float32)
    buf450 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_101(c_void_p(buf448.data_ptr()), c_void_p(buf447.data_ptr()), c_void_p(buf449.data_ptr()), c_void_p(buf450.data_ptr()))
    buf451 = reinterpret_tensor(buf443, (16, 128, 64), (8192, 64, 1), 0); del buf443  # reuse
    # Source Nodes: [attn_output_80], Original ATen: [aten.bmm]
    extern_kernels.bmm(buf450, reinterpret_tensor(buf444, (16, 128, 64), (8192, 64, 1), 0), out=buf451)
    buf452 = buf440; del buf440  # reuse
    cpp_fused_view_102(c_void_p(buf451.data_ptr()), c_void_p(buf452.data_ptr()))
    buf453 = reinterpret_tensor(buf451, (128, 1024), (1024, 1), 0); del buf451  # reuse
    # Source Nodes: [hidden_states_168], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_258, buf452, reinterpret_tensor(primals_257, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf453)
    del primals_258
    buf454 = buf435; del buf435  # reuse
    buf455 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf457 = empty((1, 128, 1024), device='cpu', dtype=torch.float32)
    buf458 = empty((128, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_103(c_void_p(buf426.data_ptr()), c_void_p(buf434.data_ptr()), c_void_p(buf453.data_ptr()), c_void_p(primals_259.data_ptr()), c_void_p(primals_260.data_ptr()), c_void_p(buf454.data_ptr()), c_void_p(buf455.data_ptr()), c_void_p(buf457.data_ptr()), c_void_p(buf458.data_ptr()))
    del primals_260
    buf459 = empty((128, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_decoder_layers_2_encoder_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_262, buf458, reinterpret_tensor(primals_261, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf459)
    del primals_262
    buf460 = empty((128, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_decoder_layers_2_encoder_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_264, reinterpret_tensor(buf337, (128, 1024), (1024, 1), 0), reinterpret_tensor(primals_263, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf460)
    del primals_264
    buf461 = empty((1, 16, 128, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_104(c_void_p(buf460.data_ptr()), c_void_p(buf461.data_ptr()))
    buf462 = buf460; del buf460  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_2_encoder_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_266, reinterpret_tensor(buf337, (128, 1024), (1024, 1), 0), reinterpret_tensor(primals_265, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf462)
    del primals_266
    buf463 = empty((1, 16, 128, 64), device='cpu', dtype=torch.float32)
    buf464 = empty((1, 16, 128, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_105(c_void_p(buf462.data_ptr()), c_void_p(buf459.data_ptr()), c_void_p(buf463.data_ptr()), c_void_p(buf464.data_ptr()))
    buf465 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_weights_40], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf464, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf461, (16, 64, 128), (8192, 1, 64), 0), out=buf465)
    buf466 = reinterpret_tensor(buf447, (16, 128, 1), (128, 1, 1), 0); del buf447  # reuse
    buf467 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    buf468 = empty((16, 128, 1), device='cpu', dtype=torch.float32)
    buf469 = buf467; del buf467  # reuse
    cpp_fused__softmax_106(c_void_p(buf469.data_ptr()), c_void_p(buf465.data_ptr()), c_void_p(buf466.data_ptr()), c_void_p(buf468.data_ptr()))
    buf470 = reinterpret_tensor(buf462, (16, 128, 64), (8192, 64, 1), 0); del buf462  # reuse
    # Source Nodes: [attn_output_85], Original ATen: [aten.bmm]
    extern_kernels.bmm(buf469, reinterpret_tensor(buf463, (16, 128, 64), (8192, 64, 1), 0), out=buf470)
    buf471 = buf459; del buf459  # reuse
    cpp_fused_view_107(c_void_p(buf470.data_ptr()), c_void_p(buf471.data_ptr()))
    buf472 = reinterpret_tensor(buf470, (128, 1024), (1024, 1), 0); del buf470  # reuse
    # Source Nodes: [hidden_states_172], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_268, buf471, reinterpret_tensor(primals_267, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf472)
    del primals_268
    buf473 = buf454; del buf454  # reuse
    buf474 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf476 = empty((1, 128, 1024), device='cpu', dtype=torch.float32)
    buf477 = empty((128, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_108(c_void_p(buf426.data_ptr()), c_void_p(buf434.data_ptr()), c_void_p(buf453.data_ptr()), c_void_p(buf472.data_ptr()), c_void_p(primals_269.data_ptr()), c_void_p(primals_270.data_ptr()), c_void_p(buf473.data_ptr()), c_void_p(buf474.data_ptr()), c_void_p(buf476.data_ptr()), c_void_p(buf477.data_ptr()))
    del primals_270
    buf478 = empty((128, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_decoder_layers_2_fc1], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_272, buf477, reinterpret_tensor(primals_271, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf478)
    del primals_272
    buf479 = empty((128, 4096), device='cpu', dtype=torch.float32)
    cpp_fused_relu_view_109(c_void_p(buf478.data_ptr()), c_void_p(buf479.data_ptr()))
    buf480 = empty((128, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_178], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_274, buf479, reinterpret_tensor(primals_273, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf480)
    del primals_274
    buf481 = reinterpret_tensor(buf480, (1, 128, 1024), (131072, 1024, 1), 0); del buf480  # reuse
    buf482 = buf473; del buf473  # reuse
    buf483 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf485 = empty((1, 128, 1024), device='cpu', dtype=torch.float32)
    buf486 = empty((128, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_110(c_void_p(buf481.data_ptr()), c_void_p(buf426.data_ptr()), c_void_p(buf434.data_ptr()), c_void_p(buf453.data_ptr()), c_void_p(buf472.data_ptr()), c_void_p(primals_275.data_ptr()), c_void_p(primals_276.data_ptr()), c_void_p(buf482.data_ptr()), c_void_p(buf483.data_ptr()), c_void_p(buf485.data_ptr()), c_void_p(buf486.data_ptr()))
    del primals_276
    buf487 = buf472; del buf472  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_3_self_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_278, buf486, reinterpret_tensor(primals_277, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf487)
    del primals_278
    buf488 = buf453; del buf453  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_3_self_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_280, buf486, reinterpret_tensor(primals_279, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf488)
    del primals_280
    buf489 = reinterpret_tensor(buf434, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf434  # reuse
    cpp_fused_clone_111(c_void_p(buf488.data_ptr()), c_void_p(buf489.data_ptr()))
    buf490 = buf488; del buf488  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_3_self_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_282, buf486, reinterpret_tensor(primals_281, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf490)
    del primals_282
    buf491 = reinterpret_tensor(buf426, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf426  # reuse
    buf492 = empty((1, 16, 128, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_112(c_void_p(buf490.data_ptr()), c_void_p(buf487.data_ptr()), c_void_p(buf491.data_ptr()), c_void_p(buf492.data_ptr()))
    buf493 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_weights_42], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf492, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf489, (16, 64, 128), (8192, 1, 64), 0), out=buf493)
    buf494 = empty_strided((16, 128, 1), (128, 1, 2048), device='cpu', dtype=torch.float32)
    buf495 = buf493; del buf493  # reuse
    buf496 = empty_strided((16, 128, 1), (128, 1, 2048), device='cpu', dtype=torch.float32)
    buf497 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_113(c_void_p(buf495.data_ptr()), c_void_p(buf494.data_ptr()), c_void_p(buf496.data_ptr()), c_void_p(buf497.data_ptr()))
    buf498 = reinterpret_tensor(buf490, (16, 128, 64), (8192, 64, 1), 0); del buf490  # reuse
    # Source Nodes: [attn_output_90], Original ATen: [aten.bmm]
    extern_kernels.bmm(buf497, reinterpret_tensor(buf491, (16, 128, 64), (8192, 64, 1), 0), out=buf498)
    buf499 = buf487; del buf487  # reuse
    cpp_fused_view_114(c_void_p(buf498.data_ptr()), c_void_p(buf499.data_ptr()))
    buf500 = reinterpret_tensor(buf498, (128, 1024), (1024, 1), 0); del buf498  # reuse
    # Source Nodes: [hidden_states_183], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_284, buf499, reinterpret_tensor(primals_283, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf500)
    del primals_284
    buf501 = buf482; del buf482  # reuse
    buf502 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf504 = empty((1, 128, 1024), device='cpu', dtype=torch.float32)
    buf505 = empty((128, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_115(c_void_p(buf481.data_ptr()), c_void_p(buf500.data_ptr()), c_void_p(primals_285.data_ptr()), c_void_p(primals_286.data_ptr()), c_void_p(buf501.data_ptr()), c_void_p(buf502.data_ptr()), c_void_p(buf504.data_ptr()), c_void_p(buf505.data_ptr()))
    del primals_286
    buf506 = empty((128, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_decoder_layers_3_encoder_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_288, buf505, reinterpret_tensor(primals_287, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf506)
    del primals_288
    buf507 = empty((128, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_decoder_layers_3_encoder_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_290, reinterpret_tensor(buf337, (128, 1024), (1024, 1), 0), reinterpret_tensor(primals_289, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf507)
    del primals_290
    buf508 = empty((1, 16, 128, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_116(c_void_p(buf507.data_ptr()), c_void_p(buf508.data_ptr()))
    buf509 = buf507; del buf507  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_3_encoder_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_292, reinterpret_tensor(buf337, (128, 1024), (1024, 1), 0), reinterpret_tensor(primals_291, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf509)
    del primals_292
    buf510 = empty((1, 16, 128, 64), device='cpu', dtype=torch.float32)
    buf511 = empty((1, 16, 128, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_117(c_void_p(buf509.data_ptr()), c_void_p(buf506.data_ptr()), c_void_p(buf510.data_ptr()), c_void_p(buf511.data_ptr()))
    buf512 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_weights_46], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf511, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf508, (16, 64, 128), (8192, 1, 64), 0), out=buf512)
    buf513 = reinterpret_tensor(buf494, (16, 128, 1), (128, 1, 1), 0); del buf494  # reuse
    buf514 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    buf515 = empty((16, 128, 1), device='cpu', dtype=torch.float32)
    buf516 = buf514; del buf514  # reuse
    cpp_fused__softmax_118(c_void_p(buf516.data_ptr()), c_void_p(buf512.data_ptr()), c_void_p(buf513.data_ptr()), c_void_p(buf515.data_ptr()))
    buf517 = reinterpret_tensor(buf509, (16, 128, 64), (8192, 64, 1), 0); del buf509  # reuse
    # Source Nodes: [attn_output_95], Original ATen: [aten.bmm]
    extern_kernels.bmm(buf516, reinterpret_tensor(buf510, (16, 128, 64), (8192, 64, 1), 0), out=buf517)
    buf518 = buf506; del buf506  # reuse
    cpp_fused_view_119(c_void_p(buf517.data_ptr()), c_void_p(buf518.data_ptr()))
    buf519 = reinterpret_tensor(buf517, (128, 1024), (1024, 1), 0); del buf517  # reuse
    # Source Nodes: [hidden_states_187], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_294, buf518, reinterpret_tensor(primals_293, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf519)
    del primals_294
    buf520 = buf501; del buf501  # reuse
    buf521 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf523 = empty((1, 128, 1024), device='cpu', dtype=torch.float32)
    buf524 = empty((128, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_120(c_void_p(buf481.data_ptr()), c_void_p(buf500.data_ptr()), c_void_p(buf519.data_ptr()), c_void_p(primals_295.data_ptr()), c_void_p(primals_296.data_ptr()), c_void_p(buf520.data_ptr()), c_void_p(buf521.data_ptr()), c_void_p(buf523.data_ptr()), c_void_p(buf524.data_ptr()))
    del primals_296
    buf525 = empty((128, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_decoder_layers_3_fc1], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_298, buf524, reinterpret_tensor(primals_297, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf525)
    del primals_298
    buf526 = empty((128, 4096), device='cpu', dtype=torch.float32)
    cpp_fused_relu_view_121(c_void_p(buf525.data_ptr()), c_void_p(buf526.data_ptr()))
    buf527 = empty((128, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_193], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_300, buf526, reinterpret_tensor(primals_299, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf527)
    del primals_300
    buf528 = buf520; del buf520  # reuse
    buf529 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf531 = empty((1, 128, 1024), device='cpu', dtype=torch.float32)
    buf532 = empty((128, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_122(c_void_p(buf481.data_ptr()), c_void_p(buf500.data_ptr()), c_void_p(buf519.data_ptr()), c_void_p(buf527.data_ptr()), c_void_p(primals_301.data_ptr()), c_void_p(primals_302.data_ptr()), c_void_p(buf528.data_ptr()), c_void_p(buf529.data_ptr()), c_void_p(buf531.data_ptr()), c_void_p(buf532.data_ptr()))
    del primals_302
    buf533 = empty((128, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_decoder_layers_4_self_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_304, buf532, reinterpret_tensor(primals_303, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf533)
    del primals_304
    buf534 = empty((128, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_decoder_layers_4_self_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_306, buf532, reinterpret_tensor(primals_305, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf534)
    del primals_306
    buf535 = empty((1, 16, 128, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_123(c_void_p(buf534.data_ptr()), c_void_p(buf535.data_ptr()))
    buf536 = buf534; del buf534  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_4_self_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_308, buf532, reinterpret_tensor(primals_307, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf536)
    del primals_308
    buf537 = empty((1, 16, 128, 64), device='cpu', dtype=torch.float32)
    buf538 = empty((1, 16, 128, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_124(c_void_p(buf536.data_ptr()), c_void_p(buf533.data_ptr()), c_void_p(buf537.data_ptr()), c_void_p(buf538.data_ptr()))
    buf539 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_weights_48], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf538, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf535, (16, 64, 128), (8192, 1, 64), 0), out=buf539)
    buf540 = empty_strided((16, 128, 1), (128, 1, 2048), device='cpu', dtype=torch.float32)
    buf541 = buf539; del buf539  # reuse
    buf542 = empty_strided((16, 128, 1), (128, 1, 2048), device='cpu', dtype=torch.float32)
    buf543 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_125(c_void_p(buf541.data_ptr()), c_void_p(buf540.data_ptr()), c_void_p(buf542.data_ptr()), c_void_p(buf543.data_ptr()))
    buf544 = reinterpret_tensor(buf536, (16, 128, 64), (8192, 64, 1), 0); del buf536  # reuse
    # Source Nodes: [attn_output_100], Original ATen: [aten.bmm]
    extern_kernels.bmm(buf543, reinterpret_tensor(buf537, (16, 128, 64), (8192, 64, 1), 0), out=buf544)
    buf545 = buf533; del buf533  # reuse
    cpp_fused_view_126(c_void_p(buf544.data_ptr()), c_void_p(buf545.data_ptr()))
    buf546 = reinterpret_tensor(buf544, (128, 1024), (1024, 1), 0); del buf544  # reuse
    # Source Nodes: [hidden_states_198], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_310, buf545, reinterpret_tensor(primals_309, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf546)
    del primals_310
    buf547 = reinterpret_tensor(buf546, (1, 128, 1024), (131072, 1024, 1), 0); del buf546  # reuse
    buf548 = buf528; del buf528  # reuse
    buf549 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf551 = empty((1, 128, 1024), device='cpu', dtype=torch.float32)
    buf552 = empty((128, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_127(c_void_p(buf547.data_ptr()), c_void_p(buf481.data_ptr()), c_void_p(buf500.data_ptr()), c_void_p(buf519.data_ptr()), c_void_p(buf527.data_ptr()), c_void_p(primals_311.data_ptr()), c_void_p(primals_312.data_ptr()), c_void_p(buf548.data_ptr()), c_void_p(buf549.data_ptr()), c_void_p(buf551.data_ptr()), c_void_p(buf552.data_ptr()))
    del primals_312
    buf553 = buf527; del buf527  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_4_encoder_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_314, buf552, reinterpret_tensor(primals_313, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf553)
    del primals_314
    buf554 = buf519; del buf519  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_4_encoder_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_316, reinterpret_tensor(buf337, (128, 1024), (1024, 1), 0), reinterpret_tensor(primals_315, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf554)
    del primals_316
    buf555 = reinterpret_tensor(buf500, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf500  # reuse
    cpp_fused_clone_128(c_void_p(buf554.data_ptr()), c_void_p(buf555.data_ptr()))
    buf556 = buf554; del buf554  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_4_encoder_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_318, reinterpret_tensor(buf337, (128, 1024), (1024, 1), 0), reinterpret_tensor(primals_317, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf556)
    del primals_318
    buf557 = reinterpret_tensor(buf481, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf481  # reuse
    buf558 = empty((1, 16, 128, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_129(c_void_p(buf556.data_ptr()), c_void_p(buf553.data_ptr()), c_void_p(buf557.data_ptr()), c_void_p(buf558.data_ptr()))
    buf559 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_weights_52], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf558, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf555, (16, 64, 128), (8192, 1, 64), 0), out=buf559)
    buf560 = reinterpret_tensor(buf540, (16, 128, 1), (128, 1, 1), 0); del buf540  # reuse
    buf561 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    buf562 = empty((16, 128, 1), device='cpu', dtype=torch.float32)
    buf563 = buf561; del buf561  # reuse
    cpp_fused__softmax_130(c_void_p(buf563.data_ptr()), c_void_p(buf559.data_ptr()), c_void_p(buf560.data_ptr()), c_void_p(buf562.data_ptr()))
    buf564 = reinterpret_tensor(buf556, (16, 128, 64), (8192, 64, 1), 0); del buf556  # reuse
    # Source Nodes: [attn_output_105], Original ATen: [aten.bmm]
    extern_kernels.bmm(buf563, reinterpret_tensor(buf557, (16, 128, 64), (8192, 64, 1), 0), out=buf564)
    buf565 = buf553; del buf553  # reuse
    cpp_fused_view_131(c_void_p(buf564.data_ptr()), c_void_p(buf565.data_ptr()))
    buf566 = reinterpret_tensor(buf564, (128, 1024), (1024, 1), 0); del buf564  # reuse
    # Source Nodes: [hidden_states_202], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_320, buf565, reinterpret_tensor(primals_319, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf566)
    del primals_320
    buf567 = buf548; del buf548  # reuse
    buf568 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf570 = empty((1, 128, 1024), device='cpu', dtype=torch.float32)
    buf571 = empty((128, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_132(c_void_p(buf547.data_ptr()), c_void_p(buf566.data_ptr()), c_void_p(primals_321.data_ptr()), c_void_p(primals_322.data_ptr()), c_void_p(buf567.data_ptr()), c_void_p(buf568.data_ptr()), c_void_p(buf570.data_ptr()), c_void_p(buf571.data_ptr()))
    del primals_322
    buf572 = empty((128, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_decoder_layers_4_fc1], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_324, buf571, reinterpret_tensor(primals_323, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf572)
    del primals_324
    buf573 = empty((128, 4096), device='cpu', dtype=torch.float32)
    cpp_fused_relu_view_133(c_void_p(buf572.data_ptr()), c_void_p(buf573.data_ptr()))
    buf574 = empty((128, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_208], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_326, buf573, reinterpret_tensor(primals_325, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf574)
    del primals_326
    buf575 = buf567; del buf567  # reuse
    buf576 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf578 = empty((1, 128, 1024), device='cpu', dtype=torch.float32)
    buf579 = empty((128, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_134(c_void_p(buf547.data_ptr()), c_void_p(buf566.data_ptr()), c_void_p(buf574.data_ptr()), c_void_p(primals_327.data_ptr()), c_void_p(primals_328.data_ptr()), c_void_p(buf575.data_ptr()), c_void_p(buf576.data_ptr()), c_void_p(buf578.data_ptr()), c_void_p(buf579.data_ptr()))
    del primals_328
    buf580 = empty((128, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_decoder_layers_5_self_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_330, buf579, reinterpret_tensor(primals_329, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf580)
    del primals_330
    buf581 = empty((128, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_decoder_layers_5_self_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_332, buf579, reinterpret_tensor(primals_331, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf581)
    del primals_332
    buf582 = empty((1, 16, 128, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_135(c_void_p(buf581.data_ptr()), c_void_p(buf582.data_ptr()))
    buf583 = buf581; del buf581  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_5_self_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_334, buf579, reinterpret_tensor(primals_333, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf583)
    del primals_334
    buf584 = empty((1, 16, 128, 64), device='cpu', dtype=torch.float32)
    buf585 = empty((1, 16, 128, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_136(c_void_p(buf583.data_ptr()), c_void_p(buf580.data_ptr()), c_void_p(buf584.data_ptr()), c_void_p(buf585.data_ptr()))
    buf586 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_weights_54], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf585, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf582, (16, 64, 128), (8192, 1, 64), 0), out=buf586)
    buf587 = empty_strided((16, 128, 1), (128, 1, 2048), device='cpu', dtype=torch.float32)
    buf588 = buf586; del buf586  # reuse
    buf589 = empty_strided((16, 128, 1), (128, 1, 2048), device='cpu', dtype=torch.float32)
    buf590 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_137(c_void_p(buf588.data_ptr()), c_void_p(buf587.data_ptr()), c_void_p(buf589.data_ptr()), c_void_p(buf590.data_ptr()))
    buf591 = reinterpret_tensor(buf583, (16, 128, 64), (8192, 64, 1), 0); del buf583  # reuse
    # Source Nodes: [attn_output_110], Original ATen: [aten.bmm]
    extern_kernels.bmm(buf590, reinterpret_tensor(buf584, (16, 128, 64), (8192, 64, 1), 0), out=buf591)
    buf592 = buf580; del buf580  # reuse
    cpp_fused_view_138(c_void_p(buf591.data_ptr()), c_void_p(buf592.data_ptr()))
    buf593 = reinterpret_tensor(buf591, (128, 1024), (1024, 1), 0); del buf591  # reuse
    # Source Nodes: [hidden_states_213], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_336, buf592, reinterpret_tensor(primals_335, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf593)
    del primals_336
    buf594 = buf575; del buf575  # reuse
    buf595 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf597 = empty((1, 128, 1024), device='cpu', dtype=torch.float32)
    buf598 = empty((128, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_139(c_void_p(buf547.data_ptr()), c_void_p(buf566.data_ptr()), c_void_p(buf574.data_ptr()), c_void_p(buf593.data_ptr()), c_void_p(primals_337.data_ptr()), c_void_p(primals_338.data_ptr()), c_void_p(buf594.data_ptr()), c_void_p(buf595.data_ptr()), c_void_p(buf597.data_ptr()), c_void_p(buf598.data_ptr()))
    del primals_338
    buf599 = empty((128, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_decoder_layers_5_encoder_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_340, buf598, reinterpret_tensor(primals_339, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf599)
    del primals_340
    buf600 = empty((128, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_decoder_layers_5_encoder_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_342, reinterpret_tensor(buf337, (128, 1024), (1024, 1), 0), reinterpret_tensor(primals_341, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf600)
    del primals_342
    buf601 = empty((1, 16, 128, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_140(c_void_p(buf600.data_ptr()), c_void_p(buf601.data_ptr()))
    buf602 = buf600; del buf600  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_5_encoder_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_344, reinterpret_tensor(buf337, (128, 1024), (1024, 1), 0), reinterpret_tensor(primals_343, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf602)
    del primals_344
    buf603 = empty((1, 16, 128, 64), device='cpu', dtype=torch.float32)
    buf604 = empty((1, 16, 128, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_141(c_void_p(buf602.data_ptr()), c_void_p(buf599.data_ptr()), c_void_p(buf603.data_ptr()), c_void_p(buf604.data_ptr()))
    buf605 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_weights_58], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf604, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf601, (16, 64, 128), (8192, 1, 64), 0), out=buf605)
    buf606 = reinterpret_tensor(buf587, (16, 128, 1), (128, 1, 1), 0); del buf587  # reuse
    buf607 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    buf608 = empty((16, 128, 1), device='cpu', dtype=torch.float32)
    buf609 = buf607; del buf607  # reuse
    cpp_fused__softmax_142(c_void_p(buf609.data_ptr()), c_void_p(buf605.data_ptr()), c_void_p(buf606.data_ptr()), c_void_p(buf608.data_ptr()))
    buf610 = reinterpret_tensor(buf602, (16, 128, 64), (8192, 64, 1), 0); del buf602  # reuse
    # Source Nodes: [attn_output_115], Original ATen: [aten.bmm]
    extern_kernels.bmm(buf609, reinterpret_tensor(buf603, (16, 128, 64), (8192, 64, 1), 0), out=buf610)
    buf611 = buf599; del buf599  # reuse
    cpp_fused_view_143(c_void_p(buf610.data_ptr()), c_void_p(buf611.data_ptr()))
    buf612 = reinterpret_tensor(buf610, (128, 1024), (1024, 1), 0); del buf610  # reuse
    # Source Nodes: [hidden_states_217], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_346, buf611, reinterpret_tensor(primals_345, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf612)
    del primals_346
    buf613 = reinterpret_tensor(buf612, (1, 128, 1024), (131072, 1024, 1), 0); del buf612  # reuse
    buf614 = buf594; del buf594  # reuse
    buf615 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf617 = empty((1, 128, 1024), device='cpu', dtype=torch.float32)
    buf618 = empty((128, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_144(c_void_p(buf613.data_ptr()), c_void_p(buf547.data_ptr()), c_void_p(buf566.data_ptr()), c_void_p(buf574.data_ptr()), c_void_p(buf593.data_ptr()), c_void_p(primals_347.data_ptr()), c_void_p(primals_348.data_ptr()), c_void_p(buf614.data_ptr()), c_void_p(buf615.data_ptr()), c_void_p(buf617.data_ptr()), c_void_p(buf618.data_ptr()))
    del primals_348
    buf619 = empty((128, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_decoder_layers_5_fc1], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_350, buf618, reinterpret_tensor(primals_349, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf619)
    del primals_350
    buf620 = empty((128, 4096), device='cpu', dtype=torch.float32)
    cpp_fused_relu_view_145(c_void_p(buf619.data_ptr()), c_void_p(buf620.data_ptr()))
    buf621 = buf593; del buf593  # reuse
    # Source Nodes: [hidden_states_223], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_352, buf620, reinterpret_tensor(primals_351, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf621)
    del primals_352
    buf622 = buf614; del buf614  # reuse
    buf623 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf625 = reinterpret_tensor(buf574, (1, 128, 1024), (131072, 1024, 1), 0); del buf574  # reuse
    buf626 = buf566; del buf566  # reuse
    cpp_fused_add_native_layer_norm_view_146(c_void_p(buf613.data_ptr()), c_void_p(buf621.data_ptr()), c_void_p(primals_353.data_ptr()), c_void_p(primals_354.data_ptr()), c_void_p(buf622.data_ptr()), c_void_p(buf623.data_ptr()), c_void_p(buf625.data_ptr()), c_void_p(buf626.data_ptr()))
    del primals_354
    buf627 = reinterpret_tensor(buf547, (128, 1024), (1024, 1), 0); del buf547  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_6_self_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_356, buf626, reinterpret_tensor(primals_355, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf627)
    del primals_356
    buf628 = empty((128, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_decoder_layers_6_self_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_358, buf626, reinterpret_tensor(primals_357, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf628)
    del primals_358
    buf629 = empty((1, 16, 128, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_147(c_void_p(buf628.data_ptr()), c_void_p(buf629.data_ptr()))
    buf630 = buf628; del buf628  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_6_self_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_360, buf626, reinterpret_tensor(primals_359, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf630)
    del primals_360
    buf631 = empty((1, 16, 128, 64), device='cpu', dtype=torch.float32)
    buf632 = empty((1, 16, 128, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_148(c_void_p(buf630.data_ptr()), c_void_p(buf627.data_ptr()), c_void_p(buf631.data_ptr()), c_void_p(buf632.data_ptr()))
    buf633 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_weights_60], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf632, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf629, (16, 64, 128), (8192, 1, 64), 0), out=buf633)
    buf634 = empty_strided((16, 128, 1), (128, 1, 2048), device='cpu', dtype=torch.float32)
    buf635 = buf633; del buf633  # reuse
    buf636 = empty_strided((16, 128, 1), (128, 1, 2048), device='cpu', dtype=torch.float32)
    buf637 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_149(c_void_p(buf635.data_ptr()), c_void_p(buf634.data_ptr()), c_void_p(buf636.data_ptr()), c_void_p(buf637.data_ptr()))
    buf638 = reinterpret_tensor(buf630, (16, 128, 64), (8192, 64, 1), 0); del buf630  # reuse
    # Source Nodes: [attn_output_120], Original ATen: [aten.bmm]
    extern_kernels.bmm(buf637, reinterpret_tensor(buf631, (16, 128, 64), (8192, 64, 1), 0), out=buf638)
    buf639 = buf627; del buf627  # reuse
    cpp_fused_view_150(c_void_p(buf638.data_ptr()), c_void_p(buf639.data_ptr()))
    buf640 = reinterpret_tensor(buf638, (128, 1024), (1024, 1), 0); del buf638  # reuse
    # Source Nodes: [hidden_states_228], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_362, buf639, reinterpret_tensor(primals_361, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf640)
    del primals_362
    buf641 = buf622; del buf622  # reuse
    buf642 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf644 = empty((1, 128, 1024), device='cpu', dtype=torch.float32)
    buf645 = empty((128, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_151(c_void_p(buf613.data_ptr()), c_void_p(buf621.data_ptr()), c_void_p(buf640.data_ptr()), c_void_p(primals_363.data_ptr()), c_void_p(primals_364.data_ptr()), c_void_p(buf641.data_ptr()), c_void_p(buf642.data_ptr()), c_void_p(buf644.data_ptr()), c_void_p(buf645.data_ptr()))
    del primals_364
    buf646 = empty((128, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_decoder_layers_6_encoder_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_366, buf645, reinterpret_tensor(primals_365, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf646)
    del primals_366
    buf647 = empty((128, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_decoder_layers_6_encoder_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_368, reinterpret_tensor(buf337, (128, 1024), (1024, 1), 0), reinterpret_tensor(primals_367, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf647)
    del primals_368
    buf648 = empty((1, 16, 128, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_152(c_void_p(buf647.data_ptr()), c_void_p(buf648.data_ptr()))
    buf649 = buf647; del buf647  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_6_encoder_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_370, reinterpret_tensor(buf337, (128, 1024), (1024, 1), 0), reinterpret_tensor(primals_369, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf649)
    del primals_370
    buf650 = empty((1, 16, 128, 64), device='cpu', dtype=torch.float32)
    buf651 = empty((1, 16, 128, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_153(c_void_p(buf649.data_ptr()), c_void_p(buf646.data_ptr()), c_void_p(buf650.data_ptr()), c_void_p(buf651.data_ptr()))
    buf652 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_weights_64], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf651, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf648, (16, 64, 128), (8192, 1, 64), 0), out=buf652)
    buf653 = reinterpret_tensor(buf634, (16, 128, 1), (128, 1, 1), 0); del buf634  # reuse
    buf654 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    buf655 = empty((16, 128, 1), device='cpu', dtype=torch.float32)
    buf656 = buf654; del buf654  # reuse
    cpp_fused__softmax_154(c_void_p(buf656.data_ptr()), c_void_p(buf652.data_ptr()), c_void_p(buf653.data_ptr()), c_void_p(buf655.data_ptr()))
    buf657 = reinterpret_tensor(buf649, (16, 128, 64), (8192, 64, 1), 0); del buf649  # reuse
    # Source Nodes: [attn_output_125], Original ATen: [aten.bmm]
    extern_kernels.bmm(buf656, reinterpret_tensor(buf650, (16, 128, 64), (8192, 64, 1), 0), out=buf657)
    buf658 = buf646; del buf646  # reuse
    cpp_fused_view_155(c_void_p(buf657.data_ptr()), c_void_p(buf658.data_ptr()))
    buf659 = reinterpret_tensor(buf657, (128, 1024), (1024, 1), 0); del buf657  # reuse
    # Source Nodes: [hidden_states_232], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_372, buf658, reinterpret_tensor(primals_371, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf659)
    del primals_372
    buf660 = buf641; del buf641  # reuse
    buf661 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf663 = empty((1, 128, 1024), device='cpu', dtype=torch.float32)
    buf664 = empty((128, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_156(c_void_p(buf613.data_ptr()), c_void_p(buf621.data_ptr()), c_void_p(buf640.data_ptr()), c_void_p(buf659.data_ptr()), c_void_p(primals_373.data_ptr()), c_void_p(primals_374.data_ptr()), c_void_p(buf660.data_ptr()), c_void_p(buf661.data_ptr()), c_void_p(buf663.data_ptr()), c_void_p(buf664.data_ptr()))
    del primals_374
    buf665 = empty((128, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_decoder_layers_6_fc1], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_376, buf664, reinterpret_tensor(primals_375, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf665)
    del primals_376
    buf666 = empty((128, 4096), device='cpu', dtype=torch.float32)
    cpp_fused_relu_view_157(c_void_p(buf665.data_ptr()), c_void_p(buf666.data_ptr()))
    buf667 = empty((128, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_238], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_378, buf666, reinterpret_tensor(primals_377, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf667)
    del primals_378
    buf668 = reinterpret_tensor(buf667, (1, 128, 1024), (131072, 1024, 1), 0); del buf667  # reuse
    buf669 = buf660; del buf660  # reuse
    buf670 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf672 = empty((1, 128, 1024), device='cpu', dtype=torch.float32)
    buf673 = empty((128, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_158(c_void_p(buf668.data_ptr()), c_void_p(buf613.data_ptr()), c_void_p(buf621.data_ptr()), c_void_p(buf640.data_ptr()), c_void_p(buf659.data_ptr()), c_void_p(primals_379.data_ptr()), c_void_p(primals_380.data_ptr()), c_void_p(buf669.data_ptr()), c_void_p(buf670.data_ptr()), c_void_p(buf672.data_ptr()), c_void_p(buf673.data_ptr()))
    del primals_380
    buf674 = buf659; del buf659  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_7_self_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_382, buf673, reinterpret_tensor(primals_381, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf674)
    del primals_382
    buf675 = buf640; del buf640  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_7_self_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_384, buf673, reinterpret_tensor(primals_383, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf675)
    del primals_384
    buf676 = reinterpret_tensor(buf621, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf621  # reuse
    cpp_fused_clone_159(c_void_p(buf675.data_ptr()), c_void_p(buf676.data_ptr()))
    buf677 = buf675; del buf675  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_7_self_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_386, buf673, reinterpret_tensor(primals_385, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf677)
    del primals_386
    buf678 = reinterpret_tensor(buf613, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf613  # reuse
    buf679 = empty((1, 16, 128, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_160(c_void_p(buf677.data_ptr()), c_void_p(buf674.data_ptr()), c_void_p(buf678.data_ptr()), c_void_p(buf679.data_ptr()))
    buf680 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_weights_66], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf679, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf676, (16, 64, 128), (8192, 1, 64), 0), out=buf680)
    buf681 = empty_strided((16, 128, 1), (128, 1, 2048), device='cpu', dtype=torch.float32)
    buf682 = buf680; del buf680  # reuse
    buf683 = empty_strided((16, 128, 1), (128, 1, 2048), device='cpu', dtype=torch.float32)
    buf684 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_161(c_void_p(buf682.data_ptr()), c_void_p(buf681.data_ptr()), c_void_p(buf683.data_ptr()), c_void_p(buf684.data_ptr()))
    buf685 = reinterpret_tensor(buf677, (16, 128, 64), (8192, 64, 1), 0); del buf677  # reuse
    # Source Nodes: [attn_output_130], Original ATen: [aten.bmm]
    extern_kernels.bmm(buf684, reinterpret_tensor(buf678, (16, 128, 64), (8192, 64, 1), 0), out=buf685)
    buf686 = buf674; del buf674  # reuse
    cpp_fused_view_162(c_void_p(buf685.data_ptr()), c_void_p(buf686.data_ptr()))
    buf687 = reinterpret_tensor(buf685, (128, 1024), (1024, 1), 0); del buf685  # reuse
    # Source Nodes: [hidden_states_243], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_388, buf686, reinterpret_tensor(primals_387, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf687)
    del primals_388
    buf688 = buf669; del buf669  # reuse
    buf689 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf691 = empty((1, 128, 1024), device='cpu', dtype=torch.float32)
    buf692 = empty((128, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_163(c_void_p(buf668.data_ptr()), c_void_p(buf687.data_ptr()), c_void_p(primals_389.data_ptr()), c_void_p(primals_390.data_ptr()), c_void_p(buf688.data_ptr()), c_void_p(buf689.data_ptr()), c_void_p(buf691.data_ptr()), c_void_p(buf692.data_ptr()))
    del primals_390
    buf693 = empty((128, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_decoder_layers_7_encoder_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_392, buf692, reinterpret_tensor(primals_391, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf693)
    del primals_392
    buf694 = empty((128, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_decoder_layers_7_encoder_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_394, reinterpret_tensor(buf337, (128, 1024), (1024, 1), 0), reinterpret_tensor(primals_393, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf694)
    del primals_394
    buf695 = empty((1, 16, 128, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_164(c_void_p(buf694.data_ptr()), c_void_p(buf695.data_ptr()))
    buf696 = buf694; del buf694  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_7_encoder_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_396, reinterpret_tensor(buf337, (128, 1024), (1024, 1), 0), reinterpret_tensor(primals_395, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf696)
    del primals_396
    buf697 = empty((1, 16, 128, 64), device='cpu', dtype=torch.float32)
    buf698 = empty((1, 16, 128, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_165(c_void_p(buf696.data_ptr()), c_void_p(buf693.data_ptr()), c_void_p(buf697.data_ptr()), c_void_p(buf698.data_ptr()))
    buf699 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_weights_70], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf698, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf695, (16, 64, 128), (8192, 1, 64), 0), out=buf699)
    buf700 = reinterpret_tensor(buf681, (16, 128, 1), (128, 1, 1), 0); del buf681  # reuse
    buf701 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    buf702 = empty((16, 128, 1), device='cpu', dtype=torch.float32)
    buf703 = buf701; del buf701  # reuse
    cpp_fused__softmax_166(c_void_p(buf703.data_ptr()), c_void_p(buf699.data_ptr()), c_void_p(buf700.data_ptr()), c_void_p(buf702.data_ptr()))
    buf704 = reinterpret_tensor(buf696, (16, 128, 64), (8192, 64, 1), 0); del buf696  # reuse
    # Source Nodes: [attn_output_135], Original ATen: [aten.bmm]
    extern_kernels.bmm(buf703, reinterpret_tensor(buf697, (16, 128, 64), (8192, 64, 1), 0), out=buf704)
    buf705 = buf693; del buf693  # reuse
    cpp_fused_view_167(c_void_p(buf704.data_ptr()), c_void_p(buf705.data_ptr()))
    buf706 = reinterpret_tensor(buf704, (128, 1024), (1024, 1), 0); del buf704  # reuse
    # Source Nodes: [hidden_states_247], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_398, buf705, reinterpret_tensor(primals_397, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf706)
    del primals_398
    buf707 = buf688; del buf688  # reuse
    buf708 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf710 = empty((1, 128, 1024), device='cpu', dtype=torch.float32)
    buf711 = empty((128, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_168(c_void_p(buf668.data_ptr()), c_void_p(buf687.data_ptr()), c_void_p(buf706.data_ptr()), c_void_p(primals_399.data_ptr()), c_void_p(primals_400.data_ptr()), c_void_p(buf707.data_ptr()), c_void_p(buf708.data_ptr()), c_void_p(buf710.data_ptr()), c_void_p(buf711.data_ptr()))
    del primals_400
    buf712 = empty((128, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_decoder_layers_7_fc1], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_402, buf711, reinterpret_tensor(primals_401, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf712)
    del primals_402
    buf713 = empty((128, 4096), device='cpu', dtype=torch.float32)
    cpp_fused_relu_view_169(c_void_p(buf712.data_ptr()), c_void_p(buf713.data_ptr()))
    buf714 = empty((128, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_253], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_404, buf713, reinterpret_tensor(primals_403, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf714)
    del primals_404
    buf715 = buf707; del buf707  # reuse
    buf716 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf718 = empty((1, 128, 1024), device='cpu', dtype=torch.float32)
    buf719 = empty((128, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_170(c_void_p(buf668.data_ptr()), c_void_p(buf687.data_ptr()), c_void_p(buf706.data_ptr()), c_void_p(buf714.data_ptr()), c_void_p(primals_405.data_ptr()), c_void_p(primals_406.data_ptr()), c_void_p(buf715.data_ptr()), c_void_p(buf716.data_ptr()), c_void_p(buf718.data_ptr()), c_void_p(buf719.data_ptr()))
    del primals_406
    buf720 = empty((128, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_decoder_layers_8_self_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_408, buf719, reinterpret_tensor(primals_407, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf720)
    del primals_408
    buf721 = empty((128, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_decoder_layers_8_self_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_410, buf719, reinterpret_tensor(primals_409, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf721)
    del primals_410
    buf722 = empty((1, 16, 128, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_171(c_void_p(buf721.data_ptr()), c_void_p(buf722.data_ptr()))
    buf723 = buf721; del buf721  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_8_self_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_412, buf719, reinterpret_tensor(primals_411, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf723)
    del primals_412
    buf724 = empty((1, 16, 128, 64), device='cpu', dtype=torch.float32)
    buf725 = empty((1, 16, 128, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_172(c_void_p(buf723.data_ptr()), c_void_p(buf720.data_ptr()), c_void_p(buf724.data_ptr()), c_void_p(buf725.data_ptr()))
    buf726 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_weights_72], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf725, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf722, (16, 64, 128), (8192, 1, 64), 0), out=buf726)
    buf727 = empty_strided((16, 128, 1), (128, 1, 2048), device='cpu', dtype=torch.float32)
    buf728 = buf726; del buf726  # reuse
    buf729 = empty_strided((16, 128, 1), (128, 1, 2048), device='cpu', dtype=torch.float32)
    buf730 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_173(c_void_p(buf728.data_ptr()), c_void_p(buf727.data_ptr()), c_void_p(buf729.data_ptr()), c_void_p(buf730.data_ptr()))
    buf731 = reinterpret_tensor(buf723, (16, 128, 64), (8192, 64, 1), 0); del buf723  # reuse
    # Source Nodes: [attn_output_140], Original ATen: [aten.bmm]
    extern_kernels.bmm(buf730, reinterpret_tensor(buf724, (16, 128, 64), (8192, 64, 1), 0), out=buf731)
    buf732 = buf720; del buf720  # reuse
    cpp_fused_view_174(c_void_p(buf731.data_ptr()), c_void_p(buf732.data_ptr()))
    buf733 = reinterpret_tensor(buf731, (128, 1024), (1024, 1), 0); del buf731  # reuse
    # Source Nodes: [hidden_states_258], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_414, buf732, reinterpret_tensor(primals_413, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf733)
    del primals_414
    buf734 = reinterpret_tensor(buf733, (1, 128, 1024), (131072, 1024, 1), 0); del buf733  # reuse
    buf735 = buf715; del buf715  # reuse
    buf736 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf738 = empty((1, 128, 1024), device='cpu', dtype=torch.float32)
    buf739 = empty((128, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_175(c_void_p(buf734.data_ptr()), c_void_p(buf668.data_ptr()), c_void_p(buf687.data_ptr()), c_void_p(buf706.data_ptr()), c_void_p(buf714.data_ptr()), c_void_p(primals_415.data_ptr()), c_void_p(primals_416.data_ptr()), c_void_p(buf735.data_ptr()), c_void_p(buf736.data_ptr()), c_void_p(buf738.data_ptr()), c_void_p(buf739.data_ptr()))
    del primals_416
    buf740 = buf714; del buf714  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_8_encoder_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_418, buf739, reinterpret_tensor(primals_417, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf740)
    del primals_418
    buf741 = buf706; del buf706  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_8_encoder_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_420, reinterpret_tensor(buf337, (128, 1024), (1024, 1), 0), reinterpret_tensor(primals_419, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf741)
    del primals_420
    buf742 = reinterpret_tensor(buf687, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf687  # reuse
    cpp_fused_clone_176(c_void_p(buf741.data_ptr()), c_void_p(buf742.data_ptr()))
    buf743 = buf741; del buf741  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_8_encoder_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_422, reinterpret_tensor(buf337, (128, 1024), (1024, 1), 0), reinterpret_tensor(primals_421, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf743)
    del primals_422
    buf744 = reinterpret_tensor(buf668, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf668  # reuse
    buf745 = empty((1, 16, 128, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_177(c_void_p(buf743.data_ptr()), c_void_p(buf740.data_ptr()), c_void_p(buf744.data_ptr()), c_void_p(buf745.data_ptr()))
    buf746 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_weights_76], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf745, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf742, (16, 64, 128), (8192, 1, 64), 0), out=buf746)
    buf747 = reinterpret_tensor(buf727, (16, 128, 1), (128, 1, 1), 0); del buf727  # reuse
    buf748 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    buf749 = empty((16, 128, 1), device='cpu', dtype=torch.float32)
    buf750 = buf748; del buf748  # reuse
    cpp_fused__softmax_178(c_void_p(buf750.data_ptr()), c_void_p(buf746.data_ptr()), c_void_p(buf747.data_ptr()), c_void_p(buf749.data_ptr()))
    buf751 = reinterpret_tensor(buf743, (16, 128, 64), (8192, 64, 1), 0); del buf743  # reuse
    # Source Nodes: [attn_output_145], Original ATen: [aten.bmm]
    extern_kernels.bmm(buf750, reinterpret_tensor(buf744, (16, 128, 64), (8192, 64, 1), 0), out=buf751)
    buf752 = buf740; del buf740  # reuse
    cpp_fused_view_179(c_void_p(buf751.data_ptr()), c_void_p(buf752.data_ptr()))
    buf753 = reinterpret_tensor(buf751, (128, 1024), (1024, 1), 0); del buf751  # reuse
    # Source Nodes: [hidden_states_262], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_424, buf752, reinterpret_tensor(primals_423, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf753)
    del primals_424
    buf754 = buf735; del buf735  # reuse
    buf755 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf757 = empty((1, 128, 1024), device='cpu', dtype=torch.float32)
    buf758 = empty((128, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_180(c_void_p(buf734.data_ptr()), c_void_p(buf753.data_ptr()), c_void_p(primals_425.data_ptr()), c_void_p(primals_426.data_ptr()), c_void_p(buf754.data_ptr()), c_void_p(buf755.data_ptr()), c_void_p(buf757.data_ptr()), c_void_p(buf758.data_ptr()))
    del primals_426
    buf759 = empty((128, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_decoder_layers_8_fc1], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_428, buf758, reinterpret_tensor(primals_427, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf759)
    del primals_428
    buf760 = empty((128, 4096), device='cpu', dtype=torch.float32)
    cpp_fused_relu_view_181(c_void_p(buf759.data_ptr()), c_void_p(buf760.data_ptr()))
    buf761 = empty((128, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_268], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_430, buf760, reinterpret_tensor(primals_429, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf761)
    del primals_430
    buf762 = buf754; del buf754  # reuse
    buf763 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf765 = empty((1, 128, 1024), device='cpu', dtype=torch.float32)
    buf766 = empty((128, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_182(c_void_p(buf734.data_ptr()), c_void_p(buf753.data_ptr()), c_void_p(buf761.data_ptr()), c_void_p(primals_431.data_ptr()), c_void_p(primals_432.data_ptr()), c_void_p(buf762.data_ptr()), c_void_p(buf763.data_ptr()), c_void_p(buf765.data_ptr()), c_void_p(buf766.data_ptr()))
    del primals_432
    buf767 = empty((128, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_decoder_layers_9_self_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_434, buf766, reinterpret_tensor(primals_433, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf767)
    del primals_434
    buf768 = empty((128, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_decoder_layers_9_self_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_436, buf766, reinterpret_tensor(primals_435, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf768)
    del primals_436
    buf769 = empty((1, 16, 128, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_183(c_void_p(buf768.data_ptr()), c_void_p(buf769.data_ptr()))
    buf770 = buf768; del buf768  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_9_self_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_438, buf766, reinterpret_tensor(primals_437, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf770)
    del primals_438
    buf771 = empty((1, 16, 128, 64), device='cpu', dtype=torch.float32)
    buf772 = empty((1, 16, 128, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_184(c_void_p(buf770.data_ptr()), c_void_p(buf767.data_ptr()), c_void_p(buf771.data_ptr()), c_void_p(buf772.data_ptr()))
    buf773 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_weights_78], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf772, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf769, (16, 64, 128), (8192, 1, 64), 0), out=buf773)
    buf774 = empty_strided((16, 128, 1), (128, 1, 2048), device='cpu', dtype=torch.float32)
    buf775 = buf773; del buf773  # reuse
    buf776 = empty_strided((16, 128, 1), (128, 1, 2048), device='cpu', dtype=torch.float32)
    buf777 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_185(c_void_p(buf775.data_ptr()), c_void_p(buf774.data_ptr()), c_void_p(buf776.data_ptr()), c_void_p(buf777.data_ptr()))
    buf778 = reinterpret_tensor(buf770, (16, 128, 64), (8192, 64, 1), 0); del buf770  # reuse
    # Source Nodes: [attn_output_150], Original ATen: [aten.bmm]
    extern_kernels.bmm(buf777, reinterpret_tensor(buf771, (16, 128, 64), (8192, 64, 1), 0), out=buf778)
    buf779 = buf767; del buf767  # reuse
    cpp_fused_view_186(c_void_p(buf778.data_ptr()), c_void_p(buf779.data_ptr()))
    buf780 = reinterpret_tensor(buf778, (128, 1024), (1024, 1), 0); del buf778  # reuse
    # Source Nodes: [hidden_states_273], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_440, buf779, reinterpret_tensor(primals_439, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf780)
    del primals_440
    buf781 = buf762; del buf762  # reuse
    buf782 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf784 = empty((1, 128, 1024), device='cpu', dtype=torch.float32)
    buf785 = empty((128, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_187(c_void_p(buf734.data_ptr()), c_void_p(buf753.data_ptr()), c_void_p(buf761.data_ptr()), c_void_p(buf780.data_ptr()), c_void_p(primals_441.data_ptr()), c_void_p(primals_442.data_ptr()), c_void_p(buf781.data_ptr()), c_void_p(buf782.data_ptr()), c_void_p(buf784.data_ptr()), c_void_p(buf785.data_ptr()))
    del primals_442
    buf786 = empty((128, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_decoder_layers_9_encoder_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_444, buf785, reinterpret_tensor(primals_443, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf786)
    del primals_444
    buf787 = empty((128, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_decoder_layers_9_encoder_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_446, reinterpret_tensor(buf337, (128, 1024), (1024, 1), 0), reinterpret_tensor(primals_445, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf787)
    del primals_446
    buf788 = empty((1, 16, 128, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_188(c_void_p(buf787.data_ptr()), c_void_p(buf788.data_ptr()))
    buf789 = buf787; del buf787  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_9_encoder_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_448, reinterpret_tensor(buf337, (128, 1024), (1024, 1), 0), reinterpret_tensor(primals_447, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf789)
    del primals_448
    buf790 = empty((1, 16, 128, 64), device='cpu', dtype=torch.float32)
    buf791 = empty((1, 16, 128, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_189(c_void_p(buf789.data_ptr()), c_void_p(buf786.data_ptr()), c_void_p(buf790.data_ptr()), c_void_p(buf791.data_ptr()))
    buf792 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_weights_82], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf791, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf788, (16, 64, 128), (8192, 1, 64), 0), out=buf792)
    buf793 = reinterpret_tensor(buf774, (16, 128, 1), (128, 1, 1), 0); del buf774  # reuse
    buf794 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    buf795 = empty((16, 128, 1), device='cpu', dtype=torch.float32)
    buf796 = buf794; del buf794  # reuse
    cpp_fused__softmax_190(c_void_p(buf796.data_ptr()), c_void_p(buf792.data_ptr()), c_void_p(buf793.data_ptr()), c_void_p(buf795.data_ptr()))
    buf797 = reinterpret_tensor(buf789, (16, 128, 64), (8192, 64, 1), 0); del buf789  # reuse
    # Source Nodes: [attn_output_155], Original ATen: [aten.bmm]
    extern_kernels.bmm(buf796, reinterpret_tensor(buf790, (16, 128, 64), (8192, 64, 1), 0), out=buf797)
    buf798 = buf786; del buf786  # reuse
    cpp_fused_view_191(c_void_p(buf797.data_ptr()), c_void_p(buf798.data_ptr()))
    buf799 = reinterpret_tensor(buf797, (128, 1024), (1024, 1), 0); del buf797  # reuse
    # Source Nodes: [hidden_states_277], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_450, buf798, reinterpret_tensor(primals_449, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf799)
    del primals_450
    buf800 = reinterpret_tensor(buf799, (1, 128, 1024), (131072, 1024, 1), 0); del buf799  # reuse
    buf801 = buf781; del buf781  # reuse
    buf802 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf804 = empty((1, 128, 1024), device='cpu', dtype=torch.float32)
    buf805 = empty((128, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_192(c_void_p(buf800.data_ptr()), c_void_p(buf734.data_ptr()), c_void_p(buf753.data_ptr()), c_void_p(buf761.data_ptr()), c_void_p(buf780.data_ptr()), c_void_p(primals_451.data_ptr()), c_void_p(primals_452.data_ptr()), c_void_p(buf801.data_ptr()), c_void_p(buf802.data_ptr()), c_void_p(buf804.data_ptr()), c_void_p(buf805.data_ptr()))
    del primals_452
    buf806 = empty((128, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_decoder_layers_9_fc1], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_454, buf805, reinterpret_tensor(primals_453, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf806)
    del primals_454
    buf807 = empty((128, 4096), device='cpu', dtype=torch.float32)
    cpp_fused_relu_view_193(c_void_p(buf806.data_ptr()), c_void_p(buf807.data_ptr()))
    buf808 = buf780; del buf780  # reuse
    # Source Nodes: [hidden_states_283], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_456, buf807, reinterpret_tensor(primals_455, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf808)
    del primals_456
    buf809 = buf801; del buf801  # reuse
    buf810 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf812 = reinterpret_tensor(buf761, (1, 128, 1024), (131072, 1024, 1), 0); del buf761  # reuse
    buf813 = buf753; del buf753  # reuse
    cpp_fused_add_native_layer_norm_view_194(c_void_p(buf800.data_ptr()), c_void_p(buf808.data_ptr()), c_void_p(primals_457.data_ptr()), c_void_p(primals_458.data_ptr()), c_void_p(buf809.data_ptr()), c_void_p(buf810.data_ptr()), c_void_p(buf812.data_ptr()), c_void_p(buf813.data_ptr()))
    del primals_458
    buf814 = reinterpret_tensor(buf734, (128, 1024), (1024, 1), 0); del buf734  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_10_self_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_460, buf813, reinterpret_tensor(primals_459, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf814)
    del primals_460
    buf815 = empty((128, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_decoder_layers_10_self_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_462, buf813, reinterpret_tensor(primals_461, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf815)
    del primals_462
    buf816 = empty((1, 16, 128, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_195(c_void_p(buf815.data_ptr()), c_void_p(buf816.data_ptr()))
    buf817 = buf815; del buf815  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_10_self_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_464, buf813, reinterpret_tensor(primals_463, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf817)
    del primals_464
    buf818 = empty((1, 16, 128, 64), device='cpu', dtype=torch.float32)
    buf819 = empty((1, 16, 128, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_196(c_void_p(buf817.data_ptr()), c_void_p(buf814.data_ptr()), c_void_p(buf818.data_ptr()), c_void_p(buf819.data_ptr()))
    buf820 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_weights_84], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf819, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf816, (16, 64, 128), (8192, 1, 64), 0), out=buf820)
    buf821 = empty_strided((16, 128, 1), (128, 1, 2048), device='cpu', dtype=torch.float32)
    buf822 = buf820; del buf820  # reuse
    buf823 = empty_strided((16, 128, 1), (128, 1, 2048), device='cpu', dtype=torch.float32)
    buf824 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_197(c_void_p(buf822.data_ptr()), c_void_p(buf821.data_ptr()), c_void_p(buf823.data_ptr()), c_void_p(buf824.data_ptr()))
    buf825 = reinterpret_tensor(buf817, (16, 128, 64), (8192, 64, 1), 0); del buf817  # reuse
    # Source Nodes: [attn_output_160], Original ATen: [aten.bmm]
    extern_kernels.bmm(buf824, reinterpret_tensor(buf818, (16, 128, 64), (8192, 64, 1), 0), out=buf825)
    buf826 = buf814; del buf814  # reuse
    cpp_fused_view_198(c_void_p(buf825.data_ptr()), c_void_p(buf826.data_ptr()))
    buf827 = reinterpret_tensor(buf825, (128, 1024), (1024, 1), 0); del buf825  # reuse
    # Source Nodes: [hidden_states_288], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_466, buf826, reinterpret_tensor(primals_465, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf827)
    del primals_466
    buf828 = buf809; del buf809  # reuse
    buf829 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf831 = empty((1, 128, 1024), device='cpu', dtype=torch.float32)
    buf832 = empty((128, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_199(c_void_p(buf800.data_ptr()), c_void_p(buf808.data_ptr()), c_void_p(buf827.data_ptr()), c_void_p(primals_467.data_ptr()), c_void_p(primals_468.data_ptr()), c_void_p(buf828.data_ptr()), c_void_p(buf829.data_ptr()), c_void_p(buf831.data_ptr()), c_void_p(buf832.data_ptr()))
    del primals_468
    buf833 = empty((128, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_decoder_layers_10_encoder_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_470, buf832, reinterpret_tensor(primals_469, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf833)
    del primals_470
    buf834 = empty((128, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_decoder_layers_10_encoder_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_472, reinterpret_tensor(buf337, (128, 1024), (1024, 1), 0), reinterpret_tensor(primals_471, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf834)
    del primals_472
    buf835 = empty((1, 16, 128, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_200(c_void_p(buf834.data_ptr()), c_void_p(buf835.data_ptr()))
    buf836 = buf834; del buf834  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_10_encoder_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_474, reinterpret_tensor(buf337, (128, 1024), (1024, 1), 0), reinterpret_tensor(primals_473, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf836)
    del primals_474
    buf837 = empty((1, 16, 128, 64), device='cpu', dtype=torch.float32)
    buf838 = empty((1, 16, 128, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_201(c_void_p(buf836.data_ptr()), c_void_p(buf833.data_ptr()), c_void_p(buf837.data_ptr()), c_void_p(buf838.data_ptr()))
    buf839 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_weights_88], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf838, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf835, (16, 64, 128), (8192, 1, 64), 0), out=buf839)
    buf840 = reinterpret_tensor(buf821, (16, 128, 1), (128, 1, 1), 0); del buf821  # reuse
    buf841 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    buf842 = empty((16, 128, 1), device='cpu', dtype=torch.float32)
    buf843 = buf841; del buf841  # reuse
    cpp_fused__softmax_202(c_void_p(buf843.data_ptr()), c_void_p(buf839.data_ptr()), c_void_p(buf840.data_ptr()), c_void_p(buf842.data_ptr()))
    buf844 = reinterpret_tensor(buf836, (16, 128, 64), (8192, 64, 1), 0); del buf836  # reuse
    # Source Nodes: [attn_output_165], Original ATen: [aten.bmm]
    extern_kernels.bmm(buf843, reinterpret_tensor(buf837, (16, 128, 64), (8192, 64, 1), 0), out=buf844)
    buf845 = buf833; del buf833  # reuse
    cpp_fused_view_203(c_void_p(buf844.data_ptr()), c_void_p(buf845.data_ptr()))
    buf846 = reinterpret_tensor(buf844, (128, 1024), (1024, 1), 0); del buf844  # reuse
    # Source Nodes: [hidden_states_292], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_476, buf845, reinterpret_tensor(primals_475, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf846)
    del primals_476
    buf847 = buf828; del buf828  # reuse
    buf848 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf850 = empty((1, 128, 1024), device='cpu', dtype=torch.float32)
    buf851 = empty((128, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_204(c_void_p(buf800.data_ptr()), c_void_p(buf808.data_ptr()), c_void_p(buf827.data_ptr()), c_void_p(buf846.data_ptr()), c_void_p(primals_477.data_ptr()), c_void_p(primals_478.data_ptr()), c_void_p(buf847.data_ptr()), c_void_p(buf848.data_ptr()), c_void_p(buf850.data_ptr()), c_void_p(buf851.data_ptr()))
    del primals_478
    buf852 = empty((128, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_decoder_layers_10_fc1], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_480, buf851, reinterpret_tensor(primals_479, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf852)
    del primals_480
    buf853 = empty((128, 4096), device='cpu', dtype=torch.float32)
    buf920 = empty((1, 128, 4096), device='cpu', dtype=torch.bool)
    cpp_fused_relu_threshold_backward_view_205(c_void_p(buf852.data_ptr()), c_void_p(buf853.data_ptr()), c_void_p(buf920.data_ptr()))
    buf854 = empty((128, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_298], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_482, buf853, reinterpret_tensor(primals_481, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf854)
    del primals_482
    buf855 = reinterpret_tensor(buf854, (1, 128, 1024), (131072, 1024, 1), 0); del buf854  # reuse
    buf856 = buf847; del buf847  # reuse
    buf857 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf859 = empty((1, 128, 1024), device='cpu', dtype=torch.float32)
    buf860 = empty((128, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_206(c_void_p(buf855.data_ptr()), c_void_p(buf800.data_ptr()), c_void_p(buf808.data_ptr()), c_void_p(buf827.data_ptr()), c_void_p(buf846.data_ptr()), c_void_p(primals_483.data_ptr()), c_void_p(primals_484.data_ptr()), c_void_p(buf856.data_ptr()), c_void_p(buf857.data_ptr()), c_void_p(buf859.data_ptr()), c_void_p(buf860.data_ptr()))
    del primals_484
    buf861 = buf846; del buf846  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_11_self_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_486, buf860, reinterpret_tensor(primals_485, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf861)
    del primals_486
    buf862 = buf827; del buf827  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_11_self_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_488, buf860, reinterpret_tensor(primals_487, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf862)
    del primals_488
    buf863 = reinterpret_tensor(buf808, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf808  # reuse
    cpp_fused_clone_207(c_void_p(buf862.data_ptr()), c_void_p(buf863.data_ptr()))
    buf864 = buf862; del buf862  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_11_self_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_490, buf860, reinterpret_tensor(primals_489, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf864)
    del primals_490
    buf865 = reinterpret_tensor(buf800, (1, 16, 128, 64), (131072, 8192, 64, 1), 0); del buf800  # reuse
    buf866 = empty((1, 16, 128, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_208(c_void_p(buf864.data_ptr()), c_void_p(buf861.data_ptr()), c_void_p(buf865.data_ptr()), c_void_p(buf866.data_ptr()))
    buf867 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_weights_90], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf866, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf863, (16, 64, 128), (8192, 1, 64), 0), out=buf867)
    buf868 = empty_strided((16, 128, 1), (128, 1, 2048), device='cpu', dtype=torch.float32)
    buf869 = buf867; del buf867  # reuse
    buf870 = empty_strided((16, 128, 1), (128, 1, 2048), device='cpu', dtype=torch.float32)
    buf871 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    buf918 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_detach_209(c_void_p(buf869.data_ptr()), c_void_p(buf868.data_ptr()), c_void_p(buf870.data_ptr()), c_void_p(buf871.data_ptr()), c_void_p(buf918.data_ptr()))
    buf872 = reinterpret_tensor(buf864, (16, 128, 64), (8192, 64, 1), 0); del buf864  # reuse
    # Source Nodes: [attn_output_170], Original ATen: [aten.bmm]
    extern_kernels.bmm(buf871, reinterpret_tensor(buf865, (16, 128, 64), (8192, 64, 1), 0), out=buf872)
    buf873 = buf861; del buf861  # reuse
    cpp_fused_view_210(c_void_p(buf872.data_ptr()), c_void_p(buf873.data_ptr()))
    buf874 = reinterpret_tensor(buf872, (128, 1024), (1024, 1), 0); del buf872  # reuse
    # Source Nodes: [hidden_states_303], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_492, buf873, reinterpret_tensor(primals_491, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf874)
    del primals_492
    buf875 = buf856; del buf856  # reuse
    buf876 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf878 = empty((1, 128, 1024), device='cpu', dtype=torch.float32)
    buf879 = empty((128, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_211(c_void_p(buf855.data_ptr()), c_void_p(buf874.data_ptr()), c_void_p(primals_493.data_ptr()), c_void_p(primals_494.data_ptr()), c_void_p(buf875.data_ptr()), c_void_p(buf876.data_ptr()), c_void_p(buf878.data_ptr()), c_void_p(buf879.data_ptr()))
    del primals_494
    buf880 = empty((128, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_decoder_layers_11_encoder_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_496, buf879, reinterpret_tensor(primals_495, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf880)
    del primals_496
    buf881 = empty((128, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_decoder_layers_11_encoder_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_498, reinterpret_tensor(buf337, (128, 1024), (1024, 1), 0), reinterpret_tensor(primals_497, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf881)
    del primals_498
    buf882 = empty((1, 16, 128, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_212(c_void_p(buf881.data_ptr()), c_void_p(buf882.data_ptr()))
    buf883 = buf881; del buf881  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_11_encoder_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_500, reinterpret_tensor(buf337, (128, 1024), (1024, 1), 0), reinterpret_tensor(primals_499, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf883)
    del primals_500
    buf884 = empty((1, 16, 128, 64), device='cpu', dtype=torch.float32)
    buf885 = empty((1, 16, 128, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_213(c_void_p(buf883.data_ptr()), c_void_p(buf880.data_ptr()), c_void_p(buf884.data_ptr()), c_void_p(buf885.data_ptr()))
    buf886 = buf869; del buf869  # reuse
    # Source Nodes: [attn_weights_94], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf885, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf882, (16, 64, 128), (8192, 1, 64), 0), out=buf886)
    buf887 = reinterpret_tensor(buf870, (16, 128, 1), (128, 1, 1), 0); del buf870  # reuse
    buf888 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    buf889 = reinterpret_tensor(buf868, (16, 128, 1), (128, 1, 1), 0); del buf868  # reuse
    buf890 = buf888; del buf888  # reuse
    cpp_fused__softmax_214(c_void_p(buf890.data_ptr()), c_void_p(buf886.data_ptr()), c_void_p(buf887.data_ptr()), c_void_p(buf889.data_ptr()))
    buf891 = reinterpret_tensor(buf883, (16, 128, 64), (8192, 64, 1), 0); del buf883  # reuse
    # Source Nodes: [attn_output_175], Original ATen: [aten.bmm]
    extern_kernels.bmm(buf890, reinterpret_tensor(buf884, (16, 128, 64), (8192, 64, 1), 0), out=buf891)
    buf892 = buf880; del buf880  # reuse
    cpp_fused_view_215(c_void_p(buf891.data_ptr()), c_void_p(buf892.data_ptr()))
    buf893 = reinterpret_tensor(buf891, (128, 1024), (1024, 1), 0); del buf891  # reuse
    # Source Nodes: [hidden_states_307], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_502, buf892, reinterpret_tensor(primals_501, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf893)
    del primals_502
    buf894 = buf875; del buf875  # reuse
    buf895 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf897 = empty((1, 128, 1024), device='cpu', dtype=torch.float32)
    buf898 = empty((128, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_216(c_void_p(buf855.data_ptr()), c_void_p(buf874.data_ptr()), c_void_p(buf893.data_ptr()), c_void_p(primals_503.data_ptr()), c_void_p(primals_504.data_ptr()), c_void_p(buf894.data_ptr()), c_void_p(buf895.data_ptr()), c_void_p(buf897.data_ptr()), c_void_p(buf898.data_ptr()))
    del primals_504
    buf899 = buf852; del buf852  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_11_fc1], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_506, buf898, reinterpret_tensor(primals_505, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf899)
    del primals_506
    buf900 = empty((128, 4096), device='cpu', dtype=torch.float32)
    buf915 = empty((1, 128, 4096), device='cpu', dtype=torch.bool)
    cpp_fused_relu_threshold_backward_view_217(c_void_p(buf899.data_ptr()), c_void_p(buf900.data_ptr()), c_void_p(buf915.data_ptr()))
    del buf899
    buf901 = empty((128, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_313], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_508, buf900, reinterpret_tensor(primals_507, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf901)
    del primals_508
    buf902 = buf894; del buf894  # reuse
    buf903 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf905 = empty((1, 128, 1024), device='cpu', dtype=torch.float32)
    buf906 = empty((128, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_218(c_void_p(buf855.data_ptr()), c_void_p(buf874.data_ptr()), c_void_p(buf893.data_ptr()), c_void_p(buf901.data_ptr()), c_void_p(primals_509.data_ptr()), c_void_p(primals_510.data_ptr()), c_void_p(buf902.data_ptr()), c_void_p(buf903.data_ptr()), c_void_p(buf905.data_ptr()), c_void_p(buf906.data_ptr()))
    del buf855
    del buf874
    del buf893
    del buf901
    del primals_510
    buf907 = empty((128, 128112), device='cpu', dtype=torch.float32)
    # Source Nodes: [lm_logits], Original ATen: [aten.mm]
    extern_kernels.mm(buf906, reinterpret_tensor(primals_511, (1024, 128112), (1, 1024), 0), out=buf907)
    buf908 = reinterpret_tensor(buf902, (128, 1), (1, 128), 0); del buf902  # reuse
    buf909 = empty_strided((128, 1), (1, 128), device='cpu', dtype=torch.float32)
    buf910 = empty((128, 128112), device='cpu', dtype=torch.float32)
    buf911 = empty((), device='cpu', dtype=torch.int64)
    buf913 = empty((), device='cpu', dtype=torch.float32)
    buf912 = empty((), device='cpu', dtype=torch.float32)
    buf1012 = buf913; del buf913  # reuse
    buf914 = reinterpret_tensor(buf903, (1, 128, 1), (128, 1, 1), 0); del buf903  # reuse
    buf916 = reinterpret_tensor(buf895, (1, 128, 1), (128, 1, 1), 0); del buf895  # reuse
    buf917 = reinterpret_tensor(buf876, (1, 128, 1), (128, 1, 1), 0); del buf876  # reuse
    buf919 = reinterpret_tensor(buf857, (1, 128, 1), (128, 1, 1), 0); del buf857  # reuse
    buf921 = reinterpret_tensor(buf848, (1, 128, 1), (128, 1, 1), 0); del buf848  # reuse
    buf922 = reinterpret_tensor(buf829, (1, 128, 1), (128, 1, 1), 0); del buf829  # reuse
    buf923 = buf822; del buf822  # reuse
    buf924 = reinterpret_tensor(buf810, (1, 128, 1), (128, 1, 1), 0); del buf810  # reuse
    buf925 = empty((1, 128, 4096), device='cpu', dtype=torch.bool)
    buf926 = reinterpret_tensor(buf802, (1, 128, 1), (128, 1, 1), 0); del buf802  # reuse
    buf927 = reinterpret_tensor(buf782, (1, 128, 1), (128, 1, 1), 0); del buf782  # reuse
    buf928 = buf775; del buf775  # reuse
    buf929 = reinterpret_tensor(buf763, (1, 128, 1), (128, 1, 1), 0); del buf763  # reuse
    buf930 = empty((1, 128, 4096), device='cpu', dtype=torch.bool)
    buf931 = reinterpret_tensor(buf755, (1, 128, 1), (128, 1, 1), 0); del buf755  # reuse
    buf932 = reinterpret_tensor(buf736, (1, 128, 1), (128, 1, 1), 0); del buf736  # reuse
    buf933 = buf728; del buf728  # reuse
    buf934 = reinterpret_tensor(buf716, (1, 128, 1), (128, 1, 1), 0); del buf716  # reuse
    buf935 = empty((1, 128, 4096), device='cpu', dtype=torch.bool)
    buf936 = reinterpret_tensor(buf708, (1, 128, 1), (128, 1, 1), 0); del buf708  # reuse
    buf937 = reinterpret_tensor(buf689, (1, 128, 1), (128, 1, 1), 0); del buf689  # reuse
    buf938 = buf682; del buf682  # reuse
    buf939 = reinterpret_tensor(buf670, (1, 128, 1), (128, 1, 1), 0); del buf670  # reuse
    buf940 = empty((1, 128, 4096), device='cpu', dtype=torch.bool)
    buf941 = reinterpret_tensor(buf661, (1, 128, 1), (128, 1, 1), 0); del buf661  # reuse
    buf942 = reinterpret_tensor(buf642, (1, 128, 1), (128, 1, 1), 0); del buf642  # reuse
    buf943 = buf635; del buf635  # reuse
    buf944 = reinterpret_tensor(buf623, (1, 128, 1), (128, 1, 1), 0); del buf623  # reuse
    buf945 = empty((1, 128, 4096), device='cpu', dtype=torch.bool)
    buf946 = reinterpret_tensor(buf615, (1, 128, 1), (128, 1, 1), 0); del buf615  # reuse
    buf947 = reinterpret_tensor(buf595, (1, 128, 1), (128, 1, 1), 0); del buf595  # reuse
    buf948 = buf588; del buf588  # reuse
    buf949 = reinterpret_tensor(buf576, (1, 128, 1), (128, 1, 1), 0); del buf576  # reuse
    buf950 = empty((1, 128, 4096), device='cpu', dtype=torch.bool)
    buf951 = reinterpret_tensor(buf568, (1, 128, 1), (128, 1, 1), 0); del buf568  # reuse
    buf952 = reinterpret_tensor(buf549, (1, 128, 1), (128, 1, 1), 0); del buf549  # reuse
    buf953 = buf541; del buf541  # reuse
    buf954 = reinterpret_tensor(buf529, (1, 128, 1), (128, 1, 1), 0); del buf529  # reuse
    buf955 = empty((1, 128, 4096), device='cpu', dtype=torch.bool)
    buf956 = reinterpret_tensor(buf521, (1, 128, 1), (128, 1, 1), 0); del buf521  # reuse
    buf957 = reinterpret_tensor(buf502, (1, 128, 1), (128, 1, 1), 0); del buf502  # reuse
    buf958 = buf495; del buf495  # reuse
    buf959 = reinterpret_tensor(buf483, (1, 128, 1), (128, 1, 1), 0); del buf483  # reuse
    buf960 = empty((1, 128, 4096), device='cpu', dtype=torch.bool)
    buf961 = reinterpret_tensor(buf474, (1, 128, 1), (128, 1, 1), 0); del buf474  # reuse
    buf962 = reinterpret_tensor(buf455, (1, 128, 1), (128, 1, 1), 0); del buf455  # reuse
    buf963 = buf448; del buf448  # reuse
    buf964 = reinterpret_tensor(buf436, (1, 128, 1), (128, 1, 1), 0); del buf436  # reuse
    buf965 = empty((1, 128, 4096), device='cpu', dtype=torch.bool)
    buf966 = reinterpret_tensor(buf428, (1, 128, 1), (128, 1, 1), 0); del buf428  # reuse
    buf967 = reinterpret_tensor(buf408, (1, 128, 1), (128, 1, 1), 0); del buf408  # reuse
    buf968 = buf401; del buf401  # reuse
    buf969 = reinterpret_tensor(buf389, (1, 128, 1), (128, 1, 1), 0); del buf389  # reuse
    buf970 = empty((1, 128, 4096), device='cpu', dtype=torch.bool)
    buf971 = reinterpret_tensor(buf381, (1, 128, 1), (128, 1, 1), 0); del buf381  # reuse
    buf972 = reinterpret_tensor(buf362, (1, 128, 1), (128, 1, 1), 0); del buf362  # reuse
    buf973 = buf354; del buf354  # reuse
    buf974 = reinterpret_tensor(buf342, (1, 128, 1), (128, 1, 1), 0); del buf342  # reuse
    buf975 = reinterpret_tensor(buf334, (1, 128, 1), (128, 1, 1), 0); del buf334  # reuse
    buf976 = empty((1, 128, 4096), device='cpu', dtype=torch.bool)
    buf977 = reinterpret_tensor(buf326, (1, 128, 1), (128, 1, 1), 0); del buf326  # reuse
    buf978 = reinterpret_tensor(buf307, (1, 128, 1), (128, 1, 1), 0); del buf307  # reuse
    buf979 = empty((1, 128, 4096), device='cpu', dtype=torch.bool)
    buf980 = reinterpret_tensor(buf299, (1, 128, 1), (128, 1, 1), 0); del buf299  # reuse
    buf981 = reinterpret_tensor(buf279, (1, 128, 1), (128, 1, 1), 0); del buf279  # reuse
    buf982 = empty((1, 128, 4096), device='cpu', dtype=torch.bool)
    buf983 = reinterpret_tensor(buf271, (1, 128, 1), (128, 1, 1), 0); del buf271  # reuse
    buf984 = reinterpret_tensor(buf252, (1, 128, 1), (128, 1, 1), 0); del buf252  # reuse
    buf985 = empty((1, 128, 4096), device='cpu', dtype=torch.bool)
    buf986 = reinterpret_tensor(buf244, (1, 128, 1), (128, 1, 1), 0); del buf244  # reuse
    buf987 = reinterpret_tensor(buf224, (1, 128, 1), (128, 1, 1), 0); del buf224  # reuse
    buf988 = empty((1, 128, 4096), device='cpu', dtype=torch.bool)
    buf989 = reinterpret_tensor(buf216, (1, 128, 1), (128, 1, 1), 0); del buf216  # reuse
    buf990 = reinterpret_tensor(buf197, (1, 128, 1), (128, 1, 1), 0); del buf197  # reuse
    buf991 = empty((1, 128, 4096), device='cpu', dtype=torch.bool)
    buf992 = reinterpret_tensor(buf189, (1, 128, 1), (128, 1, 1), 0); del buf189  # reuse
    buf993 = reinterpret_tensor(buf169, (1, 128, 1), (128, 1, 1), 0); del buf169  # reuse
    buf994 = empty((1, 128, 4096), device='cpu', dtype=torch.bool)
    buf995 = reinterpret_tensor(buf161, (1, 128, 1), (128, 1, 1), 0); del buf161  # reuse
    buf996 = reinterpret_tensor(buf142, (1, 128, 1), (128, 1, 1), 0); del buf142  # reuse
    buf997 = empty((1, 128, 4096), device='cpu', dtype=torch.bool)
    buf998 = reinterpret_tensor(buf134, (1, 128, 1), (128, 1, 1), 0); del buf134  # reuse
    buf999 = reinterpret_tensor(buf114, (1, 128, 1), (128, 1, 1), 0); del buf114  # reuse
    buf1000 = empty((1, 128, 4096), device='cpu', dtype=torch.bool)
    buf1001 = reinterpret_tensor(buf106, (1, 128, 1), (128, 1, 1), 0); del buf106  # reuse
    buf1002 = reinterpret_tensor(buf87, (1, 128, 1), (128, 1, 1), 0); del buf87  # reuse
    buf1003 = empty((1, 128, 4096), device='cpu', dtype=torch.bool)
    buf1004 = reinterpret_tensor(buf79, (1, 128, 1), (128, 1, 1), 0); del buf79  # reuse
    buf1005 = reinterpret_tensor(buf59, (1, 128, 1), (128, 1, 1), 0); del buf59  # reuse
    buf1006 = empty((1, 128, 4096), device='cpu', dtype=torch.bool)
    buf1007 = reinterpret_tensor(buf51, (1, 128, 1), (128, 1, 1), 0); del buf51  # reuse
    buf1008 = reinterpret_tensor(buf32, (1, 128, 1), (128, 1, 1), 0); del buf32  # reuse
    buf1009 = empty((1, 128, 4096), device='cpu', dtype=torch.bool)
    buf1010 = reinterpret_tensor(buf24, (1, 128, 1), (128, 1, 1), 0); del buf24  # reuse
    buf1011 = reinterpret_tensor(buf4, (1, 128, 1), (128, 1, 1), 0); del buf4  # reuse
    cpp_fused__log_softmax__softmax_add_detach_embedding_masked_fill_mul_native_layer_norm_native_layer_norm_backward_nll_loss_forward_relu_threshold_backward_219(c_void_p(buf1012.data_ptr()), c_void_p(buf914.data_ptr()), c_void_p(buf916.data_ptr()), c_void_p(buf917.data_ptr()), c_void_p(buf919.data_ptr()), c_void_p(buf921.data_ptr()), c_void_p(buf922.data_ptr()), c_void_p(buf923.data_ptr()), c_void_p(buf924.data_ptr()), c_void_p(buf926.data_ptr()), c_void_p(buf927.data_ptr()), c_void_p(buf928.data_ptr()), c_void_p(buf929.data_ptr()), c_void_p(buf931.data_ptr()), c_void_p(buf932.data_ptr()), c_void_p(buf933.data_ptr()), c_void_p(buf934.data_ptr()), c_void_p(buf936.data_ptr()), c_void_p(buf937.data_ptr()), c_void_p(buf938.data_ptr()), c_void_p(buf939.data_ptr()), c_void_p(buf941.data_ptr()), c_void_p(buf942.data_ptr()), c_void_p(buf943.data_ptr()), c_void_p(buf944.data_ptr()), c_void_p(buf946.data_ptr()), c_void_p(buf947.data_ptr()), c_void_p(buf948.data_ptr()), c_void_p(buf949.data_ptr()), c_void_p(buf951.data_ptr()), c_void_p(buf952.data_ptr()), c_void_p(buf953.data_ptr()), c_void_p(buf954.data_ptr()), c_void_p(buf956.data_ptr()), c_void_p(buf957.data_ptr()), c_void_p(buf958.data_ptr()), c_void_p(buf959.data_ptr()), c_void_p(buf961.data_ptr()), c_void_p(buf962.data_ptr()), c_void_p(buf963.data_ptr()), c_void_p(buf964.data_ptr()), c_void_p(buf966.data_ptr()), c_void_p(buf967.data_ptr()), c_void_p(buf968.data_ptr()), c_void_p(buf969.data_ptr()), c_void_p(buf971.data_ptr()), c_void_p(buf972.data_ptr()), c_void_p(buf973.data_ptr()), c_void_p(buf974.data_ptr()), c_void_p(buf975.data_ptr()), c_void_p(buf977.data_ptr()), c_void_p(buf978.data_ptr()), c_void_p(buf980.data_ptr()), c_void_p(buf981.data_ptr()), c_void_p(buf983.data_ptr()), c_void_p(buf984.data_ptr()), c_void_p(buf986.data_ptr()), c_void_p(buf987.data_ptr()), c_void_p(buf989.data_ptr()), c_void_p(buf990.data_ptr()), c_void_p(buf992.data_ptr()), c_void_p(buf993.data_ptr()), c_void_p(buf995.data_ptr()), c_void_p(buf996.data_ptr()), c_void_p(buf998.data_ptr()), c_void_p(buf999.data_ptr()), c_void_p(buf1001.data_ptr()), c_void_p(buf1002.data_ptr()), c_void_p(buf1004.data_ptr()), c_void_p(buf1005.data_ptr()), c_void_p(buf1007.data_ptr()), c_void_p(buf1008.data_ptr()), c_void_p(buf1010.data_ptr()), c_void_p(buf1011.data_ptr()), c_void_p(buf907.data_ptr()), c_void_p(primals_514.data_ptr()), c_void_p(buf823.data_ptr()), c_void_p(buf806.data_ptr()), c_void_p(buf776.data_ptr()), c_void_p(buf759.data_ptr()), c_void_p(buf729.data_ptr()), c_void_p(buf712.data_ptr()), c_void_p(buf683.data_ptr()), c_void_p(buf665.data_ptr()), c_void_p(buf636.data_ptr()), c_void_p(buf619.data_ptr()), c_void_p(buf589.data_ptr()), c_void_p(buf572.data_ptr()), c_void_p(buf542.data_ptr()), c_void_p(buf525.data_ptr()), c_void_p(buf496.data_ptr()), c_void_p(buf478.data_ptr()), c_void_p(buf449.data_ptr()), c_void_p(buf432.data_ptr()), c_void_p(buf402.data_ptr()), c_void_p(buf385.data_ptr()), c_void_p(buf355.data_ptr()), c_void_p(buf330.data_ptr()), c_void_p(buf303.data_ptr()), c_void_p(buf275.data_ptr()), c_void_p(buf248.data_ptr()), c_void_p(buf220.data_ptr()), c_void_p(buf193.data_ptr()), c_void_p(buf165.data_ptr()), c_void_p(buf138.data_ptr()), c_void_p(buf110.data_ptr()), c_void_p(buf83.data_ptr()), c_void_p(buf55.data_ptr()), c_void_p(buf28.data_ptr()), c_void_p(buf908.data_ptr()), c_void_p(buf909.data_ptr()), c_void_p(buf910.data_ptr()), c_void_p(buf911.data_ptr()), c_void_p(buf912.data_ptr()), c_void_p(buf925.data_ptr()), c_void_p(buf930.data_ptr()), c_void_p(buf935.data_ptr()), c_void_p(buf940.data_ptr()), c_void_p(buf945.data_ptr()), c_void_p(buf950.data_ptr()), c_void_p(buf955.data_ptr()), c_void_p(buf960.data_ptr()), c_void_p(buf965.data_ptr()), c_void_p(buf970.data_ptr()), c_void_p(buf976.data_ptr()), c_void_p(buf979.data_ptr()), c_void_p(buf982.data_ptr()), c_void_p(buf985.data_ptr()), c_void_p(buf988.data_ptr()), c_void_p(buf991.data_ptr()), c_void_p(buf994.data_ptr()), c_void_p(buf997.data_ptr()), c_void_p(buf1000.data_ptr()), c_void_p(buf1003.data_ptr()), c_void_p(buf1006.data_ptr()), c_void_p(buf1009.data_ptr()))
    return (buf1012, reinterpret_tensor(buf907, (1, 128, 128112), (16398336, 128112, 1), 0), buf348, buf350, buf368, buf370, buf395, buf397, buf414, buf416, buf442, buf444, buf461, buf463, buf489, buf491, buf508, buf510, buf535, buf537, buf555, buf557, buf582, buf584, buf601, buf603, buf629, buf631, buf648, buf650, buf676, buf678, buf695, buf697, buf722, buf724, buf742, buf744, buf769, buf771, buf788, buf790, buf816, buf818, buf835, buf837, buf863, buf865, buf882, buf884, buf337, primals_2, primals_12, primals_18, primals_28, primals_34, primals_44, primals_50, primals_60, primals_66, primals_76, primals_82, primals_92, primals_98, primals_108, primals_114, primals_124, primals_130, primals_140, primals_146, primals_156, primals_162, primals_172, primals_178, primals_188, primals_194, primals_197, primals_207, primals_217, primals_223, primals_233, primals_243, primals_249, primals_259, primals_269, primals_275, primals_285, primals_295, primals_301, primals_311, primals_321, primals_327, primals_337, primals_347, primals_353, primals_363, primals_373, primals_379, primals_389, primals_399, primals_405, primals_415, primals_425, primals_431, primals_441, primals_451, primals_457, primals_467, primals_477, primals_483, primals_493, primals_503, primals_509, primals_514, primals_516, buf6, buf7, buf13, buf14, buf16, buf20, buf26, buf27, buf29, buf34, buf35, buf41, buf42, buf44, buf48, buf53, buf54, buf56, buf61, buf62, buf68, buf69, buf71, buf75, buf81, buf82, buf84, buf89, buf90, buf96, buf97, buf99, buf103, buf108, buf109, buf111, buf116, buf117, buf123, buf124, buf126, buf130, buf136, buf137, buf139, buf144, buf145, buf151, buf152, buf154, buf158, buf163, buf164, buf166, buf171, buf172, buf178, buf179, buf181, buf185, buf191, buf192, buf194, buf199, buf200, buf206, buf207, buf209, buf213, buf218, buf219, buf221, buf226, buf227, buf233, buf234, buf236, buf240, buf246, buf247, buf249, buf254, buf255, buf261, buf262, buf264, buf268, buf273, buf274, buf276, buf281, buf282, buf288, buf289, buf291, buf295, buf301, buf302, buf304, buf309, buf310, buf316, buf317, buf319, buf323, buf328, buf329, buf331, buf336, primals_515, buf344, buf345, buf358, buf364, buf365, reinterpret_tensor(buf337, (128, 1024), (1024, 1), 0), buf372, buf373, buf375, buf378, buf383, buf384, buf386, buf391, buf392, buf405, buf410, buf411, buf418, buf419, buf421, buf424, buf430, buf431, buf433, buf438, buf439, buf452, buf457, buf458, buf465, buf466, buf468, buf471, buf476, buf477, buf479, buf485, buf486, buf499, buf504, buf505, buf512, buf513, buf515, buf518, buf523, buf524, buf526, buf531, buf532, buf545, buf551, buf552, buf559, buf560, buf562, buf565, buf570, buf571, buf573, buf578, buf579, buf592, buf597, buf598, buf605, buf606, buf608, buf611, buf617, buf618, buf620, buf625, buf626, buf639, buf644, buf645, buf652, buf653, buf655, buf658, buf663, buf664, buf666, buf672, buf673, buf686, buf691, buf692, buf699, buf700, buf702, buf705, buf710, buf711, buf713, buf718, buf719, buf732, buf738, buf739, buf746, buf747, buf749, buf752, buf757, buf758, buf760, buf765, buf766, buf779, buf784, buf785, buf792, buf793, buf795, buf798, buf804, buf805, buf807, buf812, buf813, buf826, buf831, buf832, buf839, buf840, buf842, buf845, buf850, buf851, buf853, buf859, buf860, buf873, buf878, buf879, buf886, buf887, buf889, buf892, buf897, buf898, buf900, buf905, buf906, buf910, buf912, reinterpret_tensor(primals_511, (128112, 1024), (1024, 1), 0), buf914, reinterpret_tensor(primals_507, (1024, 4096), (4096, 1), 0), buf915, reinterpret_tensor(primals_505, (4096, 1024), (1024, 1), 0), buf916, reinterpret_tensor(primals_501, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf890, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf884, (16, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf885, (16, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf882, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(primals_499, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_497, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_495, (1024, 1024), (1024, 1), 0), buf917, reinterpret_tensor(primals_491, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf871, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf865, (16, 64, 128), (8192, 1, 64), 0), buf918, reinterpret_tensor(buf866, (16, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf863, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(primals_489, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_487, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_485, (1024, 1024), (1024, 1), 0), buf919, reinterpret_tensor(primals_481, (1024, 4096), (4096, 1), 0), buf920, reinterpret_tensor(primals_479, (4096, 1024), (1024, 1), 0), buf921, reinterpret_tensor(primals_475, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf843, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf837, (16, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf838, (16, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf835, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(primals_473, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_471, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_469, (1024, 1024), (1024, 1), 0), buf922, reinterpret_tensor(primals_465, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf824, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf818, (16, 64, 128), (8192, 1, 64), 0), buf923, reinterpret_tensor(buf819, (16, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf816, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(primals_463, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_461, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_459, (1024, 1024), (1024, 1), 0), buf924, reinterpret_tensor(primals_455, (1024, 4096), (4096, 1), 0), buf925, reinterpret_tensor(primals_453, (4096, 1024), (1024, 1), 0), buf926, reinterpret_tensor(primals_449, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf796, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf790, (16, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf791, (16, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf788, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(primals_447, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_445, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_443, (1024, 1024), (1024, 1), 0), buf927, reinterpret_tensor(primals_439, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf777, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf771, (16, 64, 128), (8192, 1, 64), 0), buf928, reinterpret_tensor(buf772, (16, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf769, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(primals_437, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_435, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_433, (1024, 1024), (1024, 1), 0), buf929, reinterpret_tensor(primals_429, (1024, 4096), (4096, 1), 0), buf930, reinterpret_tensor(primals_427, (4096, 1024), (1024, 1), 0), buf931, reinterpret_tensor(primals_423, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf750, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf744, (16, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf745, (16, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf742, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(primals_421, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_419, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_417, (1024, 1024), (1024, 1), 0), buf932, reinterpret_tensor(primals_413, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf730, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf724, (16, 64, 128), (8192, 1, 64), 0), buf933, reinterpret_tensor(buf725, (16, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf722, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(primals_411, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_409, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_407, (1024, 1024), (1024, 1), 0), buf934, reinterpret_tensor(primals_403, (1024, 4096), (4096, 1), 0), buf935, reinterpret_tensor(primals_401, (4096, 1024), (1024, 1), 0), buf936, reinterpret_tensor(primals_397, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf703, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf697, (16, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf698, (16, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf695, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(primals_395, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_393, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_391, (1024, 1024), (1024, 1), 0), buf937, reinterpret_tensor(primals_387, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf684, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf678, (16, 64, 128), (8192, 1, 64), 0), buf938, reinterpret_tensor(buf679, (16, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf676, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(primals_385, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_383, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_381, (1024, 1024), (1024, 1), 0), buf939, reinterpret_tensor(primals_377, (1024, 4096), (4096, 1), 0), buf940, reinterpret_tensor(primals_375, (4096, 1024), (1024, 1), 0), buf941, reinterpret_tensor(primals_371, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf656, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf650, (16, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf651, (16, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf648, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(primals_369, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_367, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_365, (1024, 1024), (1024, 1), 0), buf942, reinterpret_tensor(primals_361, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf637, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf631, (16, 64, 128), (8192, 1, 64), 0), buf943, reinterpret_tensor(buf632, (16, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf629, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(primals_359, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_357, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_355, (1024, 1024), (1024, 1), 0), buf944, reinterpret_tensor(primals_351, (1024, 4096), (4096, 1), 0), buf945, reinterpret_tensor(primals_349, (4096, 1024), (1024, 1), 0), buf946, reinterpret_tensor(primals_345, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf609, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf603, (16, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf604, (16, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf601, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(primals_343, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_341, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_339, (1024, 1024), (1024, 1), 0), buf947, reinterpret_tensor(primals_335, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf590, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf584, (16, 64, 128), (8192, 1, 64), 0), buf948, reinterpret_tensor(buf585, (16, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf582, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(primals_333, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_331, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_329, (1024, 1024), (1024, 1), 0), buf949, reinterpret_tensor(primals_325, (1024, 4096), (4096, 1), 0), buf950, reinterpret_tensor(primals_323, (4096, 1024), (1024, 1), 0), buf951, reinterpret_tensor(primals_319, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf563, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf557, (16, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf558, (16, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf555, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(primals_317, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_315, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_313, (1024, 1024), (1024, 1), 0), buf952, reinterpret_tensor(primals_309, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf543, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf537, (16, 64, 128), (8192, 1, 64), 0), buf953, reinterpret_tensor(buf538, (16, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf535, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(primals_307, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_305, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_303, (1024, 1024), (1024, 1), 0), buf954, reinterpret_tensor(primals_299, (1024, 4096), (4096, 1), 0), buf955, reinterpret_tensor(primals_297, (4096, 1024), (1024, 1), 0), buf956, reinterpret_tensor(primals_293, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf516, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf510, (16, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf511, (16, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf508, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(primals_291, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_289, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_287, (1024, 1024), (1024, 1), 0), buf957, reinterpret_tensor(primals_283, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf497, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf491, (16, 64, 128), (8192, 1, 64), 0), buf958, reinterpret_tensor(buf492, (16, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf489, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(primals_281, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_279, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_277, (1024, 1024), (1024, 1), 0), buf959, reinterpret_tensor(primals_273, (1024, 4096), (4096, 1), 0), buf960, reinterpret_tensor(primals_271, (4096, 1024), (1024, 1), 0), buf961, reinterpret_tensor(primals_267, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf469, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf463, (16, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf464, (16, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf461, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(primals_265, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_263, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_261, (1024, 1024), (1024, 1), 0), buf962, reinterpret_tensor(primals_257, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf450, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf444, (16, 64, 128), (8192, 1, 64), 0), buf963, reinterpret_tensor(buf445, (16, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf442, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(primals_255, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_253, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_251, (1024, 1024), (1024, 1), 0), buf964, reinterpret_tensor(primals_247, (1024, 4096), (4096, 1), 0), buf965, reinterpret_tensor(primals_245, (4096, 1024), (1024, 1), 0), buf966, reinterpret_tensor(primals_241, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf422, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf416, (16, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf417, (16, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf414, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(primals_239, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_237, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_235, (1024, 1024), (1024, 1), 0), buf967, reinterpret_tensor(primals_231, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf403, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf397, (16, 64, 128), (8192, 1, 64), 0), buf968, reinterpret_tensor(buf398, (16, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf395, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(primals_229, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_227, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_225, (1024, 1024), (1024, 1), 0), buf969, reinterpret_tensor(primals_221, (1024, 4096), (4096, 1), 0), buf970, reinterpret_tensor(primals_219, (4096, 1024), (1024, 1), 0), buf971, reinterpret_tensor(primals_215, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf376, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf370, (16, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf371, (16, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf368, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(primals_213, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_211, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_209, (1024, 1024), (1024, 1), 0), buf972, reinterpret_tensor(primals_205, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf356, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf350, (16, 64, 128), (8192, 1, 64), 0), buf973, reinterpret_tensor(buf351, (16, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf348, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(primals_203, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_201, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_199, (1024, 1024), (1024, 1), 0), buf974, buf975, reinterpret_tensor(primals_192, (1024, 4096), (4096, 1), 0), buf976, reinterpret_tensor(primals_190, (4096, 1024), (1024, 1), 0), buf977, reinterpret_tensor(primals_186, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf320, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf321, (16, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf314, (16, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf315, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(primals_184, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_182, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_180, (1024, 1024), (1024, 1), 0), buf978, reinterpret_tensor(primals_176, (1024, 4096), (4096, 1), 0), buf979, reinterpret_tensor(primals_174, (4096, 1024), (1024, 1), 0), buf980, reinterpret_tensor(primals_170, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf292, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf293, (16, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf286, (16, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf287, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(primals_168, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_166, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_164, (1024, 1024), (1024, 1), 0), buf981, reinterpret_tensor(primals_160, (1024, 4096), (4096, 1), 0), buf982, reinterpret_tensor(primals_158, (4096, 1024), (1024, 1), 0), buf983, reinterpret_tensor(primals_154, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf265, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf266, (16, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf259, (16, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf260, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(primals_152, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_150, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_148, (1024, 1024), (1024, 1), 0), buf984, reinterpret_tensor(primals_144, (1024, 4096), (4096, 1), 0), buf985, reinterpret_tensor(primals_142, (4096, 1024), (1024, 1), 0), buf986, reinterpret_tensor(primals_138, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf237, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf238, (16, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf231, (16, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf232, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(primals_136, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_134, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_132, (1024, 1024), (1024, 1), 0), buf987, reinterpret_tensor(primals_128, (1024, 4096), (4096, 1), 0), buf988, reinterpret_tensor(primals_126, (4096, 1024), (1024, 1), 0), buf989, reinterpret_tensor(primals_122, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf210, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf211, (16, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf204, (16, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf205, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(primals_120, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_118, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_116, (1024, 1024), (1024, 1), 0), buf990, reinterpret_tensor(primals_112, (1024, 4096), (4096, 1), 0), buf991, reinterpret_tensor(primals_110, (4096, 1024), (1024, 1), 0), buf992, reinterpret_tensor(primals_106, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf182, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf183, (16, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf176, (16, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf177, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(primals_104, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_102, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_100, (1024, 1024), (1024, 1), 0), buf993, reinterpret_tensor(primals_96, (1024, 4096), (4096, 1), 0), buf994, reinterpret_tensor(primals_94, (4096, 1024), (1024, 1), 0), buf995, reinterpret_tensor(primals_90, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf155, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf156, (16, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf149, (16, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf150, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(primals_88, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_86, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_84, (1024, 1024), (1024, 1), 0), buf996, reinterpret_tensor(primals_80, (1024, 4096), (4096, 1), 0), buf997, reinterpret_tensor(primals_78, (4096, 1024), (1024, 1), 0), buf998, reinterpret_tensor(primals_74, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf127, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf128, (16, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf121, (16, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf122, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(primals_72, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_70, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_68, (1024, 1024), (1024, 1), 0), buf999, reinterpret_tensor(primals_64, (1024, 4096), (4096, 1), 0), buf1000, reinterpret_tensor(primals_62, (4096, 1024), (1024, 1), 0), buf1001, reinterpret_tensor(primals_58, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf100, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf101, (16, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf94, (16, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf95, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(primals_56, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_54, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_52, (1024, 1024), (1024, 1), 0), buf1002, reinterpret_tensor(primals_48, (1024, 4096), (4096, 1), 0), buf1003, reinterpret_tensor(primals_46, (4096, 1024), (1024, 1), 0), buf1004, reinterpret_tensor(primals_42, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf72, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf73, (16, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf66, (16, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf67, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(primals_40, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_38, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_36, (1024, 1024), (1024, 1), 0), buf1005, reinterpret_tensor(primals_32, (1024, 4096), (4096, 1), 0), buf1006, reinterpret_tensor(primals_30, (4096, 1024), (1024, 1), 0), buf1007, reinterpret_tensor(primals_26, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf45, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf46, (16, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf39, (16, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf40, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(primals_24, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_22, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_20, (1024, 1024), (1024, 1), 0), buf1008, reinterpret_tensor(primals_16, (1024, 4096), (4096, 1), 0), buf1009, reinterpret_tensor(primals_14, (4096, 1024), (1024, 1), 0), buf1010, reinterpret_tensor(primals_10, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf17, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf18, (16, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf11, (16, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf12, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(primals_8, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_6, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_4, (1024, 1024), (1024, 1), 0), buf1011, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((128112, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_2 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_3 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_4 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_5 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_6 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_7 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_8 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_9 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_10 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_11 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_12 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_13 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_14 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_15 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_16 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_17 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_18 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_19 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_20 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_21 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_22 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_23 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_24 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_25 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_26 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_27 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_28 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_29 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_30 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_31 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_32 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_33 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_34 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_35 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_36 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_37 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_38 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_39 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_40 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_41 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_42 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_43 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_44 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_45 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_46 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_47 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_48 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_49 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_50 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_51 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_52 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_53 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_54 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_55 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_56 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_57 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_58 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_59 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_60 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_61 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_62 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_63 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_64 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_65 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_66 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_67 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_68 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_69 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_70 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_71 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_72 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_73 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_74 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_75 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_76 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_77 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_78 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_79 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_80 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_81 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_82 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_83 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_84 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_85 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_86 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_87 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_88 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_89 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_90 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_91 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_92 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_93 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_94 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_95 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_96 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_97 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_98 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_99 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_100 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_101 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_102 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_103 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_104 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_105 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_106 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_107 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_108 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_109 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_110 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_111 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_112 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_113 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_114 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_115 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_116 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_117 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_118 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_119 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_120 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_121 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_122 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_123 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_124 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_125 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_126 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_127 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_128 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_129 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_130 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_131 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_132 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_133 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_134 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_135 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_136 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_137 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_138 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_139 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_140 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_141 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_142 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_143 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_144 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_145 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_146 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_147 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_148 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_149 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_150 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_151 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_152 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_153 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_154 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_155 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_156 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_157 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_158 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_159 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_160 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_161 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_162 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_163 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_164 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_165 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_166 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_167 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_168 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_169 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_170 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_171 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_172 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_173 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_174 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_175 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_176 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_177 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_178 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_179 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_180 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_181 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_182 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_183 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_184 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_185 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_186 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_187 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_188 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_189 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_190 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_191 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_192 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_193 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_194 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_195 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_196 = rand_strided((128112, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_197 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_198 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_199 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_200 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_201 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_202 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_203 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_204 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_205 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_206 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_207 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_208 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_209 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_210 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_211 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_212 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_213 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_214 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_215 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_216 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_217 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_218 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_219 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_220 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_221 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_222 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_223 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_224 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_225 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_226 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_227 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_228 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_229 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_230 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_231 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_232 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_233 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_234 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_235 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_236 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_237 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_238 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_239 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_240 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_241 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_242 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_243 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_244 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_245 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_246 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_247 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_248 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_249 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_250 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_251 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_252 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_253 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_254 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_255 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_256 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_257 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_258 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_259 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_260 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_261 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_262 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_263 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_264 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_265 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_266 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_267 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_268 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_269 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_270 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_271 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_272 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_273 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_274 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_275 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_276 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_277 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_278 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_279 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_280 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_281 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_282 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_283 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_284 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_285 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_286 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_287 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_288 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_289 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_290 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_291 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_292 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_293 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_294 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_295 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_296 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_297 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_298 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_299 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_300 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_301 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_302 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_303 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_304 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_305 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_306 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_307 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_308 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_309 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_310 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_311 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_312 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_313 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_314 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_315 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_316 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_317 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_318 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_319 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_320 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_321 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_322 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_323 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_324 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_325 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_326 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_327 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_328 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_329 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_330 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_331 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_332 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_333 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_334 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_335 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_336 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_337 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_338 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_339 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_340 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_341 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_342 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_343 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_344 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_345 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_346 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_347 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_348 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_349 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_350 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_351 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_352 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_353 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_354 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_355 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_356 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_357 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_358 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_359 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_360 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_361 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_362 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_363 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_364 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_365 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_366 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_367 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_368 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_369 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_370 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_371 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_372 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_373 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_374 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_375 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_376 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_377 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_378 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_379 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_380 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_381 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_382 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_383 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_384 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_385 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_386 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_387 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_388 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_389 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_390 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_391 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_392 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_393 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_394 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_395 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_396 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_397 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_398 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_399 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_400 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_401 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_402 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_403 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_404 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_405 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_406 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_407 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_408 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_409 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_410 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_411 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_412 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_413 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_414 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_415 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_416 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_417 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_418 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_419 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_420 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_421 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_422 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_423 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_424 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_425 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_426 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_427 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_428 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_429 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_430 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_431 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_432 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_433 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_434 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_435 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_436 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_437 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_438 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_439 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_440 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_441 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_442 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_443 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_444 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_445 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_446 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_447 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_448 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_449 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_450 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_451 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_452 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_453 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_454 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_455 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_456 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_457 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_458 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_459 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_460 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_461 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_462 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_463 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_464 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_465 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_466 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_467 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_468 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_469 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_470 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_471 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_472 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_473 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_474 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_475 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_476 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_477 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_478 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_479 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_480 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_481 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_482 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_483 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_484 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_485 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_486 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_487 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_488 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_489 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_490 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_491 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_492 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_493 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_494 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_495 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_496 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_497 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_498 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_499 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_500 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_501 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_502 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_503 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_504 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_505 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_506 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_507 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_508 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_509 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_510 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_511 = rand_strided((128112, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_512 = rand_strided((1026, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_513 = rand_strided((1026, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_514 = rand_strided((1, 128), (128, 1), device='cpu', dtype=torch.int64)
    primals_515 = rand_strided((1, 128), (128, 1), device='cpu', dtype=torch.int64)
    primals_516 = rand_strided((1, 128), (128, 1), device='cpu', dtype=torch.int64)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444, primals_445, primals_446, primals_447, primals_448, primals_449, primals_450, primals_451, primals_452, primals_453, primals_454, primals_455, primals_456, primals_457, primals_458, primals_459, primals_460, primals_461, primals_462, primals_463, primals_464, primals_465, primals_466, primals_467, primals_468, primals_469, primals_470, primals_471, primals_472, primals_473, primals_474, primals_475, primals_476, primals_477, primals_478, primals_479, primals_480, primals_481, primals_482, primals_483, primals_484, primals_485, primals_486, primals_487, primals_488, primals_489, primals_490, primals_491, primals_492, primals_493, primals_494, primals_495, primals_496, primals_497, primals_498, primals_499, primals_500, primals_501, primals_502, primals_503, primals_504, primals_505, primals_506, primals_507, primals_508, primals_509, primals_510, primals_511, primals_512, primals_513, primals_514, primals_515, primals_516]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('M2M100ForConditionalGeneration', benchmark_compiled_module)
