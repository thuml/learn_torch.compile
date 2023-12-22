
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(3L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x2 + (49L*x1) + (147L*x0))];
                    out_ptr0[static_cast<long>(x1 + (3L*x2) + (147L*x0))] = tmp0;
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


cpp_fused_clone_im2col_native_layer_norm_view_1 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       long* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr4)
{
    auto out_ptr3 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = c10::convert<long>(x0 + (4L*x1));
                out_ptr0[static_cast<long>(x1 + (14L*x0))] = tmp0;
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(24L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = out_ptr0[static_cast<long>((14L*(static_cast<long>(c10::div_floor_integer(((4L*(c10::div_floor_integer(x1, 4L))) + (static_cast<long>(x1) % static_cast<long>(4L))), 4L)) % static_cast<long>(4L))) + (c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(196L)), 14L)))];
                        auto tmp4 = out_ptr0[static_cast<long>((14L*(static_cast<long>(x1) % static_cast<long>(4L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(196L))) % static_cast<long>(14L)))];
                        auto tmp9 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr1[static_cast<long>(x1 + (16L*x2) + (16L*x2_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = decltype(tmp0)(tmp0 + 56);
                        auto tmp2 = tmp0 < 0;
                        auto tmp3 = tmp2 ? tmp1 : tmp0;
                        TORCH_CHECK((0 <= tmp3) & (tmp3 < 56L), "index out of bounds: 0 <= tmp3 < 56L")
                        auto tmp5 = decltype(tmp4)(tmp4 + 56);
                        auto tmp6 = tmp4 < 0;
                        auto tmp7 = tmp6 ? tmp5 : tmp4;
                        TORCH_CHECK((0 <= tmp7) & (tmp7 < 56L), "index out of bounds: 0 <= tmp7 < 56L")
                        auto tmp8 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>((24L*tmp7) + (1344L*tmp3) + (75264L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(c10::div_floor_integer(((4L*(c10::div_floor_integer(x1, 4L))) + (16L*x2) + (16L*x2_inner) + (static_cast<long>(x1) % static_cast<long>(4L))), 16L)) % static_cast<long>(24L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp10 = tmp8 + tmp9;
                        tmp10.store(out_ptr1 + static_cast<long>(x2 + (24L*x1) + (384L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (384L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr3[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(384.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = out_ptr2[static_cast<long>(x0)];
                    auto tmp4 = in_out_ptr0[static_cast<long>(x0)];
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp10 = tmp8 + tmp9;
                    tmp10.store(out_ptr4 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_cat_native_layer_norm_view_2 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       float* out_ptr0,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr5)
{
    auto out_ptr1 = in_out_ptr0;
    auto out_ptr4 = in_out_ptr1;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(384.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                    {
                        auto tmp26 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (384L*x1)));
                        auto tmp0 = c10::convert<int>(x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = masked_load(in_ptr1 + static_cast<long>(x2), to_float_mask(tmp4));
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<int>(197);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = masked_load(in_ptr0 + static_cast<long>((-384L) + x2 + (384L*x1) + (75264L*x0)), to_float_mask(tmp8));
                            auto tmp13 = out_ptr0[static_cast<long>((-1L) + x1 + (196L*x0))];
                            auto tmp14 = at::vec::Vectorized<float>(tmp13);
                            auto tmp15 = tmp12 - tmp14;
                            auto tmp16 = in_out_ptr0[static_cast<long>((-1L) + x1 + (196L*x0))];
                            auto tmp17 = at::vec::Vectorized<float>(tmp16);
                            auto tmp18 = tmp15 * tmp17;
                            auto tmp19 = masked_load(in_ptr2 + static_cast<long>(x2), to_float_mask(tmp8));
                            auto tmp20 = tmp18 * tmp19;
                            auto tmp21 = masked_load(in_ptr3 + static_cast<long>(x2), to_float_mask(tmp8));
                            auto tmp22 = tmp20 + tmp21;
                            return tmp22;
                        }
                        ;
                        auto tmp23 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                        auto tmp24 = to_float_mask(tmp4);
                        auto tmp25 = decltype(tmp7)::blendv(tmp23, tmp7, tmp24);
                        auto tmp27 = tmp25 + tmp26;
                        tmp27.store(out_ptr2 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (24L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr3[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr4[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(24.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp1 = out_ptr3[static_cast<long>(x0)];
                    auto tmp4 = in_out_ptr1[static_cast<long>(x0)];
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp10 = tmp8 + tmp9;
                    tmp10.store(out_ptr5 + static_cast<long>(x1 + (24L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_3 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                    {
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(6L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x3 + (6L*x1) + (48L*x2) + (768L*x0))];
                            out_ptr0[static_cast<long>(x3 + (6L*x2) + (96L*x1) + (384L*x0))] = tmp0;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(24L + x1 + (48L*x2) + (768L*x0)), static_cast<long>(48L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (16L*x1) + (16L*x1_inner) + (384L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_mul_4 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(100352L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (16L*x0)));
                        auto tmp1 = static_cast<float>(0.408248290463863);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp3);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(100352L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.408248290463863);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 - tmp5;
                        auto tmp7 = tmp6.exp();
                        tmp7.store(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp7;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(100352L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                    {
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(6L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr1[static_cast<long>(x3 + (6L*x1) + (24L*x2) + (384L*x0))];
                            out_ptr2[static_cast<long>(x3 + (6L*x2) + (96L*x1) + (384L*x0))] = tmp0;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_5 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((6L*(static_cast<long>(x0) % static_cast<long>(16L))) + (96L*(c10::div_floor_integer((x1 + x1_inner), 6L))) + (384L*(c10::div_floor_integer(x0, 16L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(6L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (24L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_6 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const long* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(24L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>((14L*(static_cast<long>(c10::div_floor_integer(((4L*(c10::div_floor_integer(x1, 4L))) + (static_cast<long>(x1) % static_cast<long>(4L))), 4L)) % static_cast<long>(4L))) + (c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(196L)), 14L)))];
                        auto tmp4 = in_ptr0[static_cast<long>((14L*(static_cast<long>(x1) % static_cast<long>(4L))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(196L))) % static_cast<long>(14L)))];
                        auto tmp9 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr2[static_cast<long>(x1 + (16L*x2) + (16L*x2_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp11 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (24L*x1) + (384L*x0)));
                        auto tmp1 = decltype(tmp0)(tmp0 + 56);
                        auto tmp2 = tmp0 < 0;
                        auto tmp3 = tmp2 ? tmp1 : tmp0;
                        TORCH_CHECK((0 <= tmp3) & (tmp3 < 56L), "index out of bounds: 0 <= tmp3 < 56L")
                        auto tmp5 = decltype(tmp4)(tmp4 + 56);
                        auto tmp6 = tmp4 < 0;
                        auto tmp7 = tmp6 ? tmp5 : tmp4;
                        TORCH_CHECK((0 <= tmp7) & (tmp7 < 56L), "index out of bounds: 0 <= tmp7 < 56L")
                        auto tmp8 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr1[static_cast<long>((24L*tmp7) + (1344L*tmp3) + (75264L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(c10::div_floor_integer(((4L*(c10::div_floor_integer(x1, 4L))) + (16L*x2) + (16L*x2_inner) + (static_cast<long>(x1) % static_cast<long>(4L))), 16L)) % static_cast<long>(24L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp10 = tmp8 + tmp9;
                        auto tmp12 = tmp10 + tmp11;
                        tmp12.store(in_out_ptr0 + static_cast<long>(x2 + (24L*x1) + (384L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (24L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(24.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (24L*x0)));
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (24L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_7 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2408448L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_view_8 = async_compile.cpp('''
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
                       float* out_ptr3,
                       float* out_ptr4)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (24L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (24L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(24.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp13.store(out_ptr2 + static_cast<long>(x1 + (24L*x0)));
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (24L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(static_cast<long>((x1 + x1_inner)) % static_cast<long>(24L))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp3 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr5[static_cast<long>(static_cast<long>((x1 + x1_inner)) % static_cast<long>(24L))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr4 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_cat_native_layer_norm_view_9 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = c10::convert<int>(x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2 + (75648L*x0)), to_float_mask(tmp4));
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<int>(197);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = masked_load(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp8));
                            auto tmp13 = masked_load(in_ptr1 + static_cast<long>((-384L) + x2 + (384L*x1) + (75264L*x0)), to_float_mask(tmp8));
                            auto tmp14 = tmp12 + tmp13;
                            return tmp14;
                        }
                        ;
                        auto tmp15 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                        auto tmp16 = to_float_mask(tmp4);
                        auto tmp17 = decltype(tmp7)::blendv(tmp15, tmp7, tmp16);
                        tmp17.store(out_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(384.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
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
                    tmp10.store(out_ptr3 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_10 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (768L*x2) + (151296L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (12608L*x1) + (75648L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(384L + x1 + (768L*x2) + (151296L*x0)), static_cast<long>(768L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (197L*x1) + (197L*x1_inner) + (75648L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(384L + x1 + (768L*x2) + (151296L*x0)));
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp0.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (197L*x1) + (197L*x1_inner) + (75648L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_mul_11 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9456L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (197L*x0)));
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp3);
                    }
                    #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (197L*x0))];
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp2);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9456L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 - tmp5;
                        auto tmp7 = tmp6.exp();
                        tmp7.store(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp7;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (197L*x0))];
                        auto tmp3 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 - tmp3);
                        auto tmp5 = std::exp(tmp4);
                        in_out_ptr0[static_cast<long>(x1 + (197L*x0))] = tmp5;
                        tmp_acc0 = tmp_acc0 + tmp5;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9456L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (197L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    in_out_ptr0[static_cast<long>(x1 + (197L*x0))] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x1) + (384L*x2) + (75648L*x0)));
                            tmp0.store(out_ptr2 + static_cast<long>(x3 + (64L*x2) + (12608L*x1) + (75648L*x0)));
                        }
                    }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x0) % static_cast<long>(197L))) + (12608L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (75648L*(c10::div_floor_integer(x0, 197L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
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
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(384.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp13.store(out_ptr2 + static_cast<long>(x1 + (384L*x0)));
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_14 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2420736L); x0+=static_cast<long>(8L))
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


cpp_fused_clone_15 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                    {
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(6L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x3 + (6L*x1) + (48L*x2) + (768L*x0))];
                            out_ptr0[static_cast<long>(x3 + (6L*x2) + (96L*x1) + (384L*x0))] = tmp0;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(24L + x1 + (48L*x2) + (768L*x0)), static_cast<long>(48L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (16L*x1) + (16L*x1_inner) + (384L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_mul_16 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(100352L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (16L*x0)));
                        auto tmp1 = static_cast<float>(0.408248290463863);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp3);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(100352L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.408248290463863);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 - tmp5;
                        auto tmp7 = tmp6.exp();
                        tmp7.store(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp7;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(100352L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                    {
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(6L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr1[static_cast<long>(x3 + (6L*x1) + (24L*x2) + (384L*x0))];
                            out_ptr2[static_cast<long>(x3 + (6L*x2) + (96L*x1) + (384L*x0))] = tmp0;
                        }
                    }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((6L*(static_cast<long>(x0) % static_cast<long>(16L))) + (96L*(c10::div_floor_integer((x1 + x1_inner), 6L))) + (384L*(c10::div_floor_integer(x0, 16L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(6L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (24L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (24L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (24L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (24L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(24.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-05);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (24L*x0)));
                    tmp19.store(out_ptr3 + static_cast<long>(x1 + (24L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_19 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2408448L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_view_20 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (24L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (24L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (24L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (24L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(24.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-05);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (24L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (24L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr6[static_cast<long>(static_cast<long>((x1 + x1_inner)) % static_cast<long>(24L))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp3 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr7[static_cast<long>(static_cast<long>((x1 + x1_inner)) % static_cast<long>(24L))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr4 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_cat_native_layer_norm_view_21 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr2)
{
    auto out_ptr1 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = c10::convert<int>(x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp4));
                            auto tmp7 = masked_load(in_ptr1 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp4));
                            auto tmp8 = tmp6 + tmp7;
                            auto tmp9 = masked_load(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp4));
                            auto tmp10 = tmp8 + tmp9;
                            return tmp10;
                        }
                        ;
                        auto tmp11 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp12 = tmp0 >= tmp3;
                        auto tmp13 = static_cast<int>(197);
                        auto tmp14 = tmp0 < tmp13;
                        auto tmp15 = [&]
                        {
                            auto tmp16 = masked_load(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp12));
                            auto tmp17 = masked_load(in_ptr1 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp12));
                            auto tmp18 = tmp16 + tmp17;
                            auto tmp19 = masked_load(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp12));
                            auto tmp20 = tmp18 + tmp19;
                            auto tmp21 = masked_load(in_ptr2 + static_cast<long>((-384L) + x2 + (384L*x1) + (75264L*x0)), to_float_mask(tmp12));
                            auto tmp22 = tmp20 + tmp21;
                            return tmp22;
                        }
                        ;
                        auto tmp23 = decltype(tmp15())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp15(), to_float_mask(tmp12));
                        auto tmp24 = to_float_mask(tmp4);
                        auto tmp25 = decltype(tmp11)::blendv(tmp23, tmp11, tmp24);
                        tmp25.store(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(384.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = in_out_ptr1[static_cast<long>(x0)];
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp10 = tmp8 + tmp9;
                    tmp10.store(out_ptr2 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_22 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (768L*x2) + (151296L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (12608L*x1) + (75648L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(384L + x1 + (768L*x2) + (151296L*x0)), static_cast<long>(768L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (197L*x1) + (197L*x1_inner) + (75648L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(384L + x1 + (768L*x2) + (151296L*x0)));
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp0.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (197L*x1) + (197L*x1_inner) + (75648L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_mul_23 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9456L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (197L*x0)));
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp3);
                    }
                    #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (197L*x0))];
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp2);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9456L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 - tmp5;
                        auto tmp7 = tmp6.exp();
                        tmp7.store(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp7;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (197L*x0))];
                        auto tmp3 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 - tmp3);
                        auto tmp5 = std::exp(tmp4);
                        in_out_ptr0[static_cast<long>(x1 + (197L*x0))] = tmp5;
                        tmp_acc0 = tmp_acc0 + tmp5;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9456L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (197L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    in_out_ptr0[static_cast<long>(x1 + (197L*x0))] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x1) + (384L*x2) + (75648L*x0)));
                            tmp0.store(out_ptr2 + static_cast<long>(x3 + (64L*x2) + (12608L*x1) + (75648L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_24 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x0) % static_cast<long>(197L))) + (12608L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (75648L*(c10::div_floor_integer(x0, 197L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
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
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(384.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp13.store(out_ptr2 + static_cast<long>(x1 + (384L*x0)));
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_26 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2420736L); x0+=static_cast<long>(8L))
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


cpp_fused_clone_27 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                    {
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(6L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x3 + (6L*x1) + (48L*x2) + (768L*x0))];
                            out_ptr0[static_cast<long>(x3 + (6L*x2) + (96L*x1) + (384L*x0))] = tmp0;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(24L + x1 + (48L*x2) + (768L*x0)), static_cast<long>(48L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (16L*x1) + (16L*x1_inner) + (384L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_mul_28 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(100352L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (16L*x0)));
                        auto tmp1 = static_cast<float>(0.408248290463863);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp3);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(100352L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.408248290463863);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 - tmp5;
                        auto tmp7 = tmp6.exp();
                        tmp7.store(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp7;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(100352L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                    {
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(6L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr1[static_cast<long>(x3 + (6L*x1) + (24L*x2) + (384L*x0))];
                            out_ptr2[static_cast<long>(x3 + (6L*x2) + (96L*x1) + (384L*x0))] = tmp0;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_29 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((6L*(static_cast<long>(x0) % static_cast<long>(16L))) + (96L*(c10::div_floor_integer((x1 + x1_inner), 6L))) + (384L*(c10::div_floor_integer(x0, 16L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(6L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (24L*x0)));
                }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(602112L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (24L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(24.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (24L*x0)));
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (24L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_31 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2408448L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_view_32 = async_compile.cpp('''
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
                       float* out_ptr3,
                       float* out_ptr4)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (24L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (24L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(24.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp13.store(out_ptr2 + static_cast<long>(x1 + (24L*x0)));
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (24L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(static_cast<long>((x1 + x1_inner)) % static_cast<long>(24L))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp3 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr5[static_cast<long>(static_cast<long>((x1 + x1_inner)) % static_cast<long>(24L))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr4 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_cat_native_layer_norm_view_33 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr2)
{
    auto out_ptr1 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = c10::convert<int>(x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp4));
                            auto tmp7 = masked_load(in_ptr1 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp4));
                            auto tmp8 = tmp6 + tmp7;
                            auto tmp9 = masked_load(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp4));
                            auto tmp10 = tmp8 + tmp9;
                            return tmp10;
                        }
                        ;
                        auto tmp11 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp12 = tmp0 >= tmp3;
                        auto tmp13 = static_cast<int>(197);
                        auto tmp14 = tmp0 < tmp13;
                        auto tmp15 = [&]
                        {
                            auto tmp16 = masked_load(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp12));
                            auto tmp17 = masked_load(in_ptr1 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp12));
                            auto tmp18 = tmp16 + tmp17;
                            auto tmp19 = masked_load(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp12));
                            auto tmp20 = tmp18 + tmp19;
                            auto tmp21 = masked_load(in_ptr2 + static_cast<long>((-384L) + x2 + (384L*x1) + (75264L*x0)), to_float_mask(tmp12));
                            auto tmp22 = tmp20 + tmp21;
                            return tmp22;
                        }
                        ;
                        auto tmp23 = decltype(tmp15())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp15(), to_float_mask(tmp12));
                        auto tmp24 = to_float_mask(tmp4);
                        auto tmp25 = decltype(tmp11)::blendv(tmp23, tmp11, tmp24);
                        tmp25.store(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(384.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = in_out_ptr1[static_cast<long>(x0)];
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp10 = tmp8 + tmp9;
                    tmp10.store(out_ptr2 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_34 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (768L*x2) + (151296L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (12608L*x1) + (75648L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(384L + x1 + (768L*x2) + (151296L*x0)), static_cast<long>(768L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (197L*x1) + (197L*x1_inner) + (75648L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(384L + x1 + (768L*x2) + (151296L*x0)));
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp0.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (197L*x1) + (197L*x1_inner) + (75648L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_mul_35 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9456L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (197L*x0)));
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp3);
                    }
                    #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (197L*x0))];
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp2);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9456L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 - tmp5;
                        auto tmp7 = tmp6.exp();
                        tmp7.store(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp7;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (197L*x0))];
                        auto tmp3 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 - tmp3);
                        auto tmp5 = std::exp(tmp4);
                        in_out_ptr0[static_cast<long>(x1 + (197L*x0))] = tmp5;
                        tmp_acc0 = tmp_acc0 + tmp5;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9456L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (197L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    in_out_ptr0[static_cast<long>(x1 + (197L*x0))] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x1) + (384L*x2) + (75648L*x0)));
                            tmp0.store(out_ptr2 + static_cast<long>(x3 + (64L*x2) + (12608L*x1) + (75648L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_36 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x0) % static_cast<long>(197L))) + (12608L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (75648L*(c10::div_floor_integer(x0, 197L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
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
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(384.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp13.store(out_ptr2 + static_cast<long>(x1 + (384L*x0)));
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_38 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2420736L); x0+=static_cast<long>(8L))
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


cpp_fused_clone_39 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                    {
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(6L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x3 + (6L*x1) + (48L*x2) + (768L*x0))];
                            out_ptr0[static_cast<long>(x3 + (6L*x2) + (96L*x1) + (384L*x0))] = tmp0;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(24L + x1 + (48L*x2) + (768L*x0)), static_cast<long>(48L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (16L*x1) + (16L*x1_inner) + (384L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_mul_40 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(100352L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (16L*x0)));
                        auto tmp1 = static_cast<float>(0.408248290463863);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp3);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(100352L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.408248290463863);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 - tmp5;
                        auto tmp7 = tmp6.exp();
                        tmp7.store(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp7;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(100352L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                    {
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(6L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr1[static_cast<long>(x3 + (6L*x1) + (24L*x2) + (384L*x0))];
                            out_ptr2[static_cast<long>(x3 + (6L*x2) + (96L*x1) + (384L*x0))] = tmp0;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_41 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((6L*(static_cast<long>(x0) % static_cast<long>(16L))) + (96L*(c10::div_floor_integer((x1 + x1_inner), 6L))) + (384L*(c10::div_floor_integer(x0, 16L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(6L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (24L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_42 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (24L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (24L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (24L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(24.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-05);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (24L*x0)));
                    tmp19.store(out_ptr3 + static_cast<long>(x1 + (24L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_43 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2408448L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_view_44 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (24L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (24L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (24L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (24L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(24.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-05);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (24L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (24L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr6[static_cast<long>(static_cast<long>((x1 + x1_inner)) % static_cast<long>(24L))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp3 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr7[static_cast<long>(static_cast<long>((x1 + x1_inner)) % static_cast<long>(24L))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr4 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_cat_native_layer_norm_view_45 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr2)
{
    auto out_ptr1 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = c10::convert<int>(x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp4));
                            auto tmp7 = masked_load(in_ptr1 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp4));
                            auto tmp8 = tmp6 + tmp7;
                            auto tmp9 = masked_load(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp4));
                            auto tmp10 = tmp8 + tmp9;
                            return tmp10;
                        }
                        ;
                        auto tmp11 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp12 = tmp0 >= tmp3;
                        auto tmp13 = static_cast<int>(197);
                        auto tmp14 = tmp0 < tmp13;
                        auto tmp15 = [&]
                        {
                            auto tmp16 = masked_load(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp12));
                            auto tmp17 = masked_load(in_ptr1 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp12));
                            auto tmp18 = tmp16 + tmp17;
                            auto tmp19 = masked_load(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp12));
                            auto tmp20 = tmp18 + tmp19;
                            auto tmp21 = masked_load(in_ptr2 + static_cast<long>((-384L) + x2 + (384L*x1) + (75264L*x0)), to_float_mask(tmp12));
                            auto tmp22 = tmp20 + tmp21;
                            return tmp22;
                        }
                        ;
                        auto tmp23 = decltype(tmp15())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp15(), to_float_mask(tmp12));
                        auto tmp24 = to_float_mask(tmp4);
                        auto tmp25 = decltype(tmp11)::blendv(tmp23, tmp11, tmp24);
                        tmp25.store(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(384.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = in_out_ptr1[static_cast<long>(x0)];
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp10 = tmp8 + tmp9;
                    tmp10.store(out_ptr2 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_46 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (768L*x2) + (151296L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (12608L*x1) + (75648L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(384L + x1 + (768L*x2) + (151296L*x0)), static_cast<long>(768L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (197L*x1) + (197L*x1_inner) + (75648L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(384L + x1 + (768L*x2) + (151296L*x0)));
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp0.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (197L*x1) + (197L*x1_inner) + (75648L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_mul_47 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9456L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (197L*x0)));
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp3);
                    }
                    #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (197L*x0))];
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp2);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9456L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 - tmp5;
                        auto tmp7 = tmp6.exp();
                        tmp7.store(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp7;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (197L*x0))];
                        auto tmp3 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 - tmp3);
                        auto tmp5 = std::exp(tmp4);
                        in_out_ptr0[static_cast<long>(x1 + (197L*x0))] = tmp5;
                        tmp_acc0 = tmp_acc0 + tmp5;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9456L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (197L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    in_out_ptr0[static_cast<long>(x1 + (197L*x0))] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x1) + (384L*x2) + (75648L*x0)));
                            tmp0.store(out_ptr2 + static_cast<long>(x3 + (64L*x2) + (12608L*x1) + (75648L*x0)));
                        }
                    }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x0) % static_cast<long>(197L))) + (12608L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (75648L*(c10::div_floor_integer(x0, 197L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
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
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(384.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp13.store(out_ptr2 + static_cast<long>(x1 + (384L*x0)));
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_50 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2420736L); x0+=static_cast<long>(8L))
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


cpp_fused_clone_51 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                    {
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(6L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x3 + (6L*x1) + (48L*x2) + (768L*x0))];
                            out_ptr0[static_cast<long>(x3 + (6L*x2) + (96L*x1) + (384L*x0))] = tmp0;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(24L + x1 + (48L*x2) + (768L*x0)), static_cast<long>(48L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (16L*x1) + (16L*x1_inner) + (384L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_mul_52 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(100352L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (16L*x0)));
                        auto tmp1 = static_cast<float>(0.408248290463863);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp3);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(100352L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.408248290463863);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 - tmp5;
                        auto tmp7 = tmp6.exp();
                        tmp7.store(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp7;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(100352L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                    {
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(6L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr1[static_cast<long>(x3 + (6L*x1) + (24L*x2) + (384L*x0))];
                            out_ptr2[static_cast<long>(x3 + (6L*x2) + (96L*x1) + (384L*x0))] = tmp0;
                        }
                    }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((6L*(static_cast<long>(x0) % static_cast<long>(16L))) + (96L*(c10::div_floor_integer((x1 + x1_inner), 6L))) + (384L*(c10::div_floor_integer(x0, 16L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(6L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (24L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_54 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(602112L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (24L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(24.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (24L*x0)));
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (24L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_55 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2408448L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_view_56 = async_compile.cpp('''
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
                       float* out_ptr3,
                       float* out_ptr4)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (24L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (24L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(24.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp13.store(out_ptr2 + static_cast<long>(x1 + (24L*x0)));
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (24L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(static_cast<long>((x1 + x1_inner)) % static_cast<long>(24L))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp3 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr5[static_cast<long>(static_cast<long>((x1 + x1_inner)) % static_cast<long>(24L))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr4 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_cat_native_layer_norm_view_57 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr2)
{
    auto out_ptr1 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = c10::convert<int>(x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp4));
                            auto tmp7 = masked_load(in_ptr1 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp4));
                            auto tmp8 = tmp6 + tmp7;
                            auto tmp9 = masked_load(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp4));
                            auto tmp10 = tmp8 + tmp9;
                            return tmp10;
                        }
                        ;
                        auto tmp11 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp12 = tmp0 >= tmp3;
                        auto tmp13 = static_cast<int>(197);
                        auto tmp14 = tmp0 < tmp13;
                        auto tmp15 = [&]
                        {
                            auto tmp16 = masked_load(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp12));
                            auto tmp17 = masked_load(in_ptr1 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp12));
                            auto tmp18 = tmp16 + tmp17;
                            auto tmp19 = masked_load(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp12));
                            auto tmp20 = tmp18 + tmp19;
                            auto tmp21 = masked_load(in_ptr2 + static_cast<long>((-384L) + x2 + (384L*x1) + (75264L*x0)), to_float_mask(tmp12));
                            auto tmp22 = tmp20 + tmp21;
                            return tmp22;
                        }
                        ;
                        auto tmp23 = decltype(tmp15())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp15(), to_float_mask(tmp12));
                        auto tmp24 = to_float_mask(tmp4);
                        auto tmp25 = decltype(tmp11)::blendv(tmp23, tmp11, tmp24);
                        tmp25.store(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(384.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = in_out_ptr1[static_cast<long>(x0)];
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp10 = tmp8 + tmp9;
                    tmp10.store(out_ptr2 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_58 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (768L*x2) + (151296L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (12608L*x1) + (75648L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(384L + x1 + (768L*x2) + (151296L*x0)), static_cast<long>(768L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (197L*x1) + (197L*x1_inner) + (75648L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(384L + x1 + (768L*x2) + (151296L*x0)));
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp0.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (197L*x1) + (197L*x1_inner) + (75648L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_mul_59 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9456L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (197L*x0)));
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp3);
                    }
                    #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (197L*x0))];
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp2);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9456L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 - tmp5;
                        auto tmp7 = tmp6.exp();
                        tmp7.store(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp7;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (197L*x0))];
                        auto tmp3 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 - tmp3);
                        auto tmp5 = std::exp(tmp4);
                        in_out_ptr0[static_cast<long>(x1 + (197L*x0))] = tmp5;
                        tmp_acc0 = tmp_acc0 + tmp5;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9456L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (197L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    in_out_ptr0[static_cast<long>(x1 + (197L*x0))] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x1) + (384L*x2) + (75648L*x0)));
                            tmp0.store(out_ptr2 + static_cast<long>(x3 + (64L*x2) + (12608L*x1) + (75648L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_60 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x0) % static_cast<long>(197L))) + (12608L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (75648L*(c10::div_floor_integer(x0, 197L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
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
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(384.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp13.store(out_ptr2 + static_cast<long>(x1 + (384L*x0)));
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_62 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2420736L); x0+=static_cast<long>(8L))
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


cpp_fused_clone_63 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                    {
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(6L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x3 + (6L*x1) + (48L*x2) + (768L*x0))];
                            out_ptr0[static_cast<long>(x3 + (6L*x2) + (96L*x1) + (384L*x0))] = tmp0;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(24L + x1 + (48L*x2) + (768L*x0)), static_cast<long>(48L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (16L*x1) + (16L*x1_inner) + (384L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_mul_64 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(100352L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (16L*x0)));
                        auto tmp1 = static_cast<float>(0.408248290463863);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp3);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(100352L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.408248290463863);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 - tmp5;
                        auto tmp7 = tmp6.exp();
                        tmp7.store(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp7;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(100352L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                    {
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(6L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr1[static_cast<long>(x3 + (6L*x1) + (24L*x2) + (384L*x0))];
                            out_ptr2[static_cast<long>(x3 + (6L*x2) + (96L*x1) + (384L*x0))] = tmp0;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_65 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((6L*(static_cast<long>(x0) % static_cast<long>(16L))) + (96L*(c10::div_floor_integer((x1 + x1_inner), 6L))) + (384L*(c10::div_floor_integer(x0, 16L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(6L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (24L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_66 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (24L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (24L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (24L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(24.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-05);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (24L*x0)));
                    tmp19.store(out_ptr3 + static_cast<long>(x1 + (24L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_67 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2408448L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_view_68 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (24L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (24L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (24L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (24L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(24.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-05);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (24L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (24L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr6[static_cast<long>(static_cast<long>((x1 + x1_inner)) % static_cast<long>(24L))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp3 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr7[static_cast<long>(static_cast<long>((x1 + x1_inner)) % static_cast<long>(24L))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr4 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_cat_native_layer_norm_view_69 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr2)
{
    auto out_ptr1 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = c10::convert<int>(x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp4));
                            auto tmp7 = masked_load(in_ptr1 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp4));
                            auto tmp8 = tmp6 + tmp7;
                            auto tmp9 = masked_load(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp4));
                            auto tmp10 = tmp8 + tmp9;
                            return tmp10;
                        }
                        ;
                        auto tmp11 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp12 = tmp0 >= tmp3;
                        auto tmp13 = static_cast<int>(197);
                        auto tmp14 = tmp0 < tmp13;
                        auto tmp15 = [&]
                        {
                            auto tmp16 = masked_load(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp12));
                            auto tmp17 = masked_load(in_ptr1 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp12));
                            auto tmp18 = tmp16 + tmp17;
                            auto tmp19 = masked_load(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp12));
                            auto tmp20 = tmp18 + tmp19;
                            auto tmp21 = masked_load(in_ptr2 + static_cast<long>((-384L) + x2 + (384L*x1) + (75264L*x0)), to_float_mask(tmp12));
                            auto tmp22 = tmp20 + tmp21;
                            return tmp22;
                        }
                        ;
                        auto tmp23 = decltype(tmp15())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp15(), to_float_mask(tmp12));
                        auto tmp24 = to_float_mask(tmp4);
                        auto tmp25 = decltype(tmp11)::blendv(tmp23, tmp11, tmp24);
                        tmp25.store(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(384.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = in_out_ptr1[static_cast<long>(x0)];
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp10 = tmp8 + tmp9;
                    tmp10.store(out_ptr2 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_70 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (768L*x2) + (151296L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (12608L*x1) + (75648L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(384L + x1 + (768L*x2) + (151296L*x0)), static_cast<long>(768L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (197L*x1) + (197L*x1_inner) + (75648L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(384L + x1 + (768L*x2) + (151296L*x0)));
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp0.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (197L*x1) + (197L*x1_inner) + (75648L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_mul_71 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9456L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (197L*x0)));
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp3);
                    }
                    #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (197L*x0))];
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp2);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9456L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 - tmp5;
                        auto tmp7 = tmp6.exp();
                        tmp7.store(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp7;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (197L*x0))];
                        auto tmp3 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 - tmp3);
                        auto tmp5 = std::exp(tmp4);
                        in_out_ptr0[static_cast<long>(x1 + (197L*x0))] = tmp5;
                        tmp_acc0 = tmp_acc0 + tmp5;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9456L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (197L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    in_out_ptr0[static_cast<long>(x1 + (197L*x0))] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x1) + (384L*x2) + (75648L*x0)));
                            tmp0.store(out_ptr2 + static_cast<long>(x3 + (64L*x2) + (12608L*x1) + (75648L*x0)));
                        }
                    }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x0) % static_cast<long>(197L))) + (12608L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (75648L*(c10::div_floor_integer(x0, 197L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
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
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(384.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp13.store(out_ptr2 + static_cast<long>(x1 + (384L*x0)));
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_74 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2420736L); x0+=static_cast<long>(8L))
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


cpp_fused_clone_75 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                    {
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(6L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x3 + (6L*x1) + (48L*x2) + (768L*x0))];
                            out_ptr0[static_cast<long>(x3 + (6L*x2) + (96L*x1) + (384L*x0))] = tmp0;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(24L + x1 + (48L*x2) + (768L*x0)), static_cast<long>(48L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (16L*x1) + (16L*x1_inner) + (384L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_mul_76 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(100352L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (16L*x0)));
                        auto tmp1 = static_cast<float>(0.408248290463863);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp3);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(100352L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.408248290463863);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 - tmp5;
                        auto tmp7 = tmp6.exp();
                        tmp7.store(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp7;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(100352L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                    {
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(6L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr1[static_cast<long>(x3 + (6L*x1) + (24L*x2) + (384L*x0))];
                            out_ptr2[static_cast<long>(x3 + (6L*x2) + (96L*x1) + (384L*x0))] = tmp0;
                        }
                    }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((6L*(static_cast<long>(x0) % static_cast<long>(16L))) + (96L*(c10::div_floor_integer((x1 + x1_inner), 6L))) + (384L*(c10::div_floor_integer(x0, 16L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(6L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (24L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_78 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(602112L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (24L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(24.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (24L*x0)));
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (24L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_79 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2408448L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_view_80 = async_compile.cpp('''
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
                       float* out_ptr3,
                       float* out_ptr4)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (24L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (24L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(24.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp13.store(out_ptr2 + static_cast<long>(x1 + (24L*x0)));
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (24L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(static_cast<long>((x1 + x1_inner)) % static_cast<long>(24L))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp3 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr5[static_cast<long>(static_cast<long>((x1 + x1_inner)) % static_cast<long>(24L))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr4 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_cat_native_layer_norm_view_81 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr2)
{
    auto out_ptr1 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = c10::convert<int>(x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp4));
                            auto tmp7 = masked_load(in_ptr1 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp4));
                            auto tmp8 = tmp6 + tmp7;
                            auto tmp9 = masked_load(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp4));
                            auto tmp10 = tmp8 + tmp9;
                            return tmp10;
                        }
                        ;
                        auto tmp11 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp12 = tmp0 >= tmp3;
                        auto tmp13 = static_cast<int>(197);
                        auto tmp14 = tmp0 < tmp13;
                        auto tmp15 = [&]
                        {
                            auto tmp16 = masked_load(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp12));
                            auto tmp17 = masked_load(in_ptr1 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp12));
                            auto tmp18 = tmp16 + tmp17;
                            auto tmp19 = masked_load(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp12));
                            auto tmp20 = tmp18 + tmp19;
                            auto tmp21 = masked_load(in_ptr2 + static_cast<long>((-384L) + x2 + (384L*x1) + (75264L*x0)), to_float_mask(tmp12));
                            auto tmp22 = tmp20 + tmp21;
                            return tmp22;
                        }
                        ;
                        auto tmp23 = decltype(tmp15())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp15(), to_float_mask(tmp12));
                        auto tmp24 = to_float_mask(tmp4);
                        auto tmp25 = decltype(tmp11)::blendv(tmp23, tmp11, tmp24);
                        tmp25.store(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(384.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = in_out_ptr1[static_cast<long>(x0)];
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp10 = tmp8 + tmp9;
                    tmp10.store(out_ptr2 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_82 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (768L*x2) + (151296L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (12608L*x1) + (75648L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(384L + x1 + (768L*x2) + (151296L*x0)), static_cast<long>(768L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (197L*x1) + (197L*x1_inner) + (75648L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(384L + x1 + (768L*x2) + (151296L*x0)));
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp0.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (197L*x1) + (197L*x1_inner) + (75648L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_mul_83 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9456L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (197L*x0)));
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp3);
                    }
                    #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (197L*x0))];
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp2);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9456L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 - tmp5;
                        auto tmp7 = tmp6.exp();
                        tmp7.store(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp7;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (197L*x0))];
                        auto tmp3 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 - tmp3);
                        auto tmp5 = std::exp(tmp4);
                        in_out_ptr0[static_cast<long>(x1 + (197L*x0))] = tmp5;
                        tmp_acc0 = tmp_acc0 + tmp5;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9456L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (197L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    in_out_ptr0[static_cast<long>(x1 + (197L*x0))] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x1) + (384L*x2) + (75648L*x0)));
                            tmp0.store(out_ptr2 + static_cast<long>(x3 + (64L*x2) + (12608L*x1) + (75648L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_84 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x0) % static_cast<long>(197L))) + (12608L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (75648L*(c10::div_floor_integer(x0, 197L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
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
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(384.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp13.store(out_ptr2 + static_cast<long>(x1 + (384L*x0)));
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_86 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2420736L); x0+=static_cast<long>(8L))
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


cpp_fused_clone_87 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                    {
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(6L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x3 + (6L*x1) + (48L*x2) + (768L*x0))];
                            out_ptr0[static_cast<long>(x3 + (6L*x2) + (96L*x1) + (384L*x0))] = tmp0;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(24L + x1 + (48L*x2) + (768L*x0)), static_cast<long>(48L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (16L*x1) + (16L*x1_inner) + (384L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_mul_88 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(100352L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (16L*x0)));
                        auto tmp1 = static_cast<float>(0.408248290463863);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp3);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(100352L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.408248290463863);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 - tmp5;
                        auto tmp7 = tmp6.exp();
                        tmp7.store(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp7;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(100352L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                    {
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(6L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr1[static_cast<long>(x3 + (6L*x1) + (24L*x2) + (384L*x0))];
                            out_ptr2[static_cast<long>(x3 + (6L*x2) + (96L*x1) + (384L*x0))] = tmp0;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_89 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((6L*(static_cast<long>(x0) % static_cast<long>(16L))) + (96L*(c10::div_floor_integer((x1 + x1_inner), 6L))) + (384L*(c10::div_floor_integer(x0, 16L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(6L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (24L*x0)));
                }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (24L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (24L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (24L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(24.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-05);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (24L*x0)));
                    tmp19.store(out_ptr3 + static_cast<long>(x1 + (24L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_91 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2408448L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_view_92 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (24L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (24L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (24L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (24L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(24.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-05);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (24L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (24L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr6[static_cast<long>(static_cast<long>((x1 + x1_inner)) % static_cast<long>(24L))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp3 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr7[static_cast<long>(static_cast<long>((x1 + x1_inner)) % static_cast<long>(24L))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr4 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_cat_native_layer_norm_view_93 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr2)
{
    auto out_ptr1 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = c10::convert<int>(x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp4));
                            auto tmp7 = masked_load(in_ptr1 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp4));
                            auto tmp8 = tmp6 + tmp7;
                            auto tmp9 = masked_load(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp4));
                            auto tmp10 = tmp8 + tmp9;
                            return tmp10;
                        }
                        ;
                        auto tmp11 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp12 = tmp0 >= tmp3;
                        auto tmp13 = static_cast<int>(197);
                        auto tmp14 = tmp0 < tmp13;
                        auto tmp15 = [&]
                        {
                            auto tmp16 = masked_load(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp12));
                            auto tmp17 = masked_load(in_ptr1 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp12));
                            auto tmp18 = tmp16 + tmp17;
                            auto tmp19 = masked_load(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp12));
                            auto tmp20 = tmp18 + tmp19;
                            auto tmp21 = masked_load(in_ptr2 + static_cast<long>((-384L) + x2 + (384L*x1) + (75264L*x0)), to_float_mask(tmp12));
                            auto tmp22 = tmp20 + tmp21;
                            return tmp22;
                        }
                        ;
                        auto tmp23 = decltype(tmp15())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp15(), to_float_mask(tmp12));
                        auto tmp24 = to_float_mask(tmp4);
                        auto tmp25 = decltype(tmp11)::blendv(tmp23, tmp11, tmp24);
                        tmp25.store(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(384.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = in_out_ptr1[static_cast<long>(x0)];
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp10 = tmp8 + tmp9;
                    tmp10.store(out_ptr2 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_94 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (768L*x2) + (151296L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (12608L*x1) + (75648L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(384L + x1 + (768L*x2) + (151296L*x0)), static_cast<long>(768L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (197L*x1) + (197L*x1_inner) + (75648L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(384L + x1 + (768L*x2) + (151296L*x0)));
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp0.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (197L*x1) + (197L*x1_inner) + (75648L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_mul_95 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9456L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (197L*x0)));
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp3);
                    }
                    #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (197L*x0))];
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp2);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9456L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 - tmp5;
                        auto tmp7 = tmp6.exp();
                        tmp7.store(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp7;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (197L*x0))];
                        auto tmp3 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 - tmp3);
                        auto tmp5 = std::exp(tmp4);
                        in_out_ptr0[static_cast<long>(x1 + (197L*x0))] = tmp5;
                        tmp_acc0 = tmp_acc0 + tmp5;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9456L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (197L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    in_out_ptr0[static_cast<long>(x1 + (197L*x0))] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x1) + (384L*x2) + (75648L*x0)));
                            tmp0.store(out_ptr2 + static_cast<long>(x3 + (64L*x2) + (12608L*x1) + (75648L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_96 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x0) % static_cast<long>(197L))) + (12608L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (75648L*(c10::div_floor_integer(x0, 197L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_97 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(384.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp13.store(out_ptr2 + static_cast<long>(x1 + (384L*x0)));
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_98 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2420736L); x0+=static_cast<long>(8L))
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


cpp_fused_clone_99 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                    {
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(6L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x3 + (6L*x1) + (48L*x2) + (768L*x0))];
                            out_ptr0[static_cast<long>(x3 + (6L*x2) + (96L*x1) + (384L*x0))] = tmp0;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(24L + x1 + (48L*x2) + (768L*x0)), static_cast<long>(48L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (16L*x1) + (16L*x1_inner) + (384L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_mul_100 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(100352L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (16L*x0)));
                        auto tmp1 = static_cast<float>(0.408248290463863);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp3);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(100352L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.408248290463863);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 - tmp5;
                        auto tmp7 = tmp6.exp();
                        tmp7.store(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp7;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(100352L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                    {
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(6L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr1[static_cast<long>(x3 + (6L*x1) + (24L*x2) + (384L*x0))];
                            out_ptr2[static_cast<long>(x3 + (6L*x2) + (96L*x1) + (384L*x0))] = tmp0;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_101 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((6L*(static_cast<long>(x0) % static_cast<long>(16L))) + (96L*(c10::div_floor_integer((x1 + x1_inner), 6L))) + (384L*(c10::div_floor_integer(x0, 16L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(6L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (24L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_102 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(602112L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (24L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(24.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (24L*x0)));
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (24L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_103 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2408448L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_view_104 = async_compile.cpp('''
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
                       float* out_ptr3,
                       float* out_ptr4)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (24L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (24L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(24.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp13.store(out_ptr2 + static_cast<long>(x1 + (24L*x0)));
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (24L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(static_cast<long>((x1 + x1_inner)) % static_cast<long>(24L))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp3 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr5[static_cast<long>(static_cast<long>((x1 + x1_inner)) % static_cast<long>(24L))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr4 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_cat_native_layer_norm_view_105 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr2)
{
    auto out_ptr1 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = c10::convert<int>(x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp4));
                            auto tmp7 = masked_load(in_ptr1 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp4));
                            auto tmp8 = tmp6 + tmp7;
                            auto tmp9 = masked_load(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp4));
                            auto tmp10 = tmp8 + tmp9;
                            return tmp10;
                        }
                        ;
                        auto tmp11 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp12 = tmp0 >= tmp3;
                        auto tmp13 = static_cast<int>(197);
                        auto tmp14 = tmp0 < tmp13;
                        auto tmp15 = [&]
                        {
                            auto tmp16 = masked_load(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp12));
                            auto tmp17 = masked_load(in_ptr1 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp12));
                            auto tmp18 = tmp16 + tmp17;
                            auto tmp19 = masked_load(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp12));
                            auto tmp20 = tmp18 + tmp19;
                            auto tmp21 = masked_load(in_ptr2 + static_cast<long>((-384L) + x2 + (384L*x1) + (75264L*x0)), to_float_mask(tmp12));
                            auto tmp22 = tmp20 + tmp21;
                            return tmp22;
                        }
                        ;
                        auto tmp23 = decltype(tmp15())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp15(), to_float_mask(tmp12));
                        auto tmp24 = to_float_mask(tmp4);
                        auto tmp25 = decltype(tmp11)::blendv(tmp23, tmp11, tmp24);
                        tmp25.store(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(384.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = in_out_ptr1[static_cast<long>(x0)];
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp10 = tmp8 + tmp9;
                    tmp10.store(out_ptr2 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_106 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (768L*x2) + (151296L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (12608L*x1) + (75648L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(384L + x1 + (768L*x2) + (151296L*x0)), static_cast<long>(768L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (197L*x1) + (197L*x1_inner) + (75648L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(384L + x1 + (768L*x2) + (151296L*x0)));
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp0.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (197L*x1) + (197L*x1_inner) + (75648L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_mul_107 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9456L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (197L*x0)));
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp3);
                    }
                    #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (197L*x0))];
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp2);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9456L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 - tmp5;
                        auto tmp7 = tmp6.exp();
                        tmp7.store(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp7;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (197L*x0))];
                        auto tmp3 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 - tmp3);
                        auto tmp5 = std::exp(tmp4);
                        in_out_ptr0[static_cast<long>(x1 + (197L*x0))] = tmp5;
                        tmp_acc0 = tmp_acc0 + tmp5;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9456L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (197L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    in_out_ptr0[static_cast<long>(x1 + (197L*x0))] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x1) + (384L*x2) + (75648L*x0)));
                            tmp0.store(out_ptr2 + static_cast<long>(x3 + (64L*x2) + (12608L*x1) + (75648L*x0)));
                        }
                    }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x0) % static_cast<long>(197L))) + (12608L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (75648L*(c10::div_floor_integer(x0, 197L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_109 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(384.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp13.store(out_ptr2 + static_cast<long>(x1 + (384L*x0)));
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_110 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2420736L); x0+=static_cast<long>(8L))
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


cpp_fused_clone_111 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                    {
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(6L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x3 + (6L*x1) + (48L*x2) + (768L*x0))];
                            out_ptr0[static_cast<long>(x3 + (6L*x2) + (96L*x1) + (384L*x0))] = tmp0;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(24L + x1 + (48L*x2) + (768L*x0)), static_cast<long>(48L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (16L*x1) + (16L*x1_inner) + (384L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_mul_112 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(100352L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (16L*x0)));
                        auto tmp1 = static_cast<float>(0.408248290463863);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp3);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(100352L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.408248290463863);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 - tmp5;
                        auto tmp7 = tmp6.exp();
                        tmp7.store(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp7;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(100352L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                    {
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(6L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr1[static_cast<long>(x3 + (6L*x1) + (24L*x2) + (384L*x0))];
                            out_ptr2[static_cast<long>(x3 + (6L*x2) + (96L*x1) + (384L*x0))] = tmp0;
                        }
                    }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((6L*(static_cast<long>(x0) % static_cast<long>(16L))) + (96L*(c10::div_floor_integer((x1 + x1_inner), 6L))) + (384L*(c10::div_floor_integer(x0, 16L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(6L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (24L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_114 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (24L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (24L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (24L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(24.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-05);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (24L*x0)));
                    tmp19.store(out_ptr3 + static_cast<long>(x1 + (24L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_115 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2408448L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_view_116 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (24L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (24L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (24L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (24L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(24.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-05);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (24L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (24L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr6[static_cast<long>(static_cast<long>((x1 + x1_inner)) % static_cast<long>(24L))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp3 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr7[static_cast<long>(static_cast<long>((x1 + x1_inner)) % static_cast<long>(24L))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr4 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_cat_native_layer_norm_view_117 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr2)
{
    auto out_ptr1 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = c10::convert<int>(x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp4));
                            auto tmp7 = masked_load(in_ptr1 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp4));
                            auto tmp8 = tmp6 + tmp7;
                            auto tmp9 = masked_load(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp4));
                            auto tmp10 = tmp8 + tmp9;
                            return tmp10;
                        }
                        ;
                        auto tmp11 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp12 = tmp0 >= tmp3;
                        auto tmp13 = static_cast<int>(197);
                        auto tmp14 = tmp0 < tmp13;
                        auto tmp15 = [&]
                        {
                            auto tmp16 = masked_load(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp12));
                            auto tmp17 = masked_load(in_ptr1 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp12));
                            auto tmp18 = tmp16 + tmp17;
                            auto tmp19 = masked_load(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp12));
                            auto tmp20 = tmp18 + tmp19;
                            auto tmp21 = masked_load(in_ptr2 + static_cast<long>((-384L) + x2 + (384L*x1) + (75264L*x0)), to_float_mask(tmp12));
                            auto tmp22 = tmp20 + tmp21;
                            return tmp22;
                        }
                        ;
                        auto tmp23 = decltype(tmp15())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp15(), to_float_mask(tmp12));
                        auto tmp24 = to_float_mask(tmp4);
                        auto tmp25 = decltype(tmp11)::blendv(tmp23, tmp11, tmp24);
                        tmp25.store(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(384.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = in_out_ptr1[static_cast<long>(x0)];
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp10 = tmp8 + tmp9;
                    tmp10.store(out_ptr2 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_118 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (768L*x2) + (151296L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (12608L*x1) + (75648L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(384L + x1 + (768L*x2) + (151296L*x0)), static_cast<long>(768L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (197L*x1) + (197L*x1_inner) + (75648L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(384L + x1 + (768L*x2) + (151296L*x0)));
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp0.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (197L*x1) + (197L*x1_inner) + (75648L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_mul_119 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9456L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (197L*x0)));
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp3);
                    }
                    #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (197L*x0))];
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp2);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9456L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 - tmp5;
                        auto tmp7 = tmp6.exp();
                        tmp7.store(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp7;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (197L*x0))];
                        auto tmp3 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 - tmp3);
                        auto tmp5 = std::exp(tmp4);
                        in_out_ptr0[static_cast<long>(x1 + (197L*x0))] = tmp5;
                        tmp_acc0 = tmp_acc0 + tmp5;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9456L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (197L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    in_out_ptr0[static_cast<long>(x1 + (197L*x0))] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x1) + (384L*x2) + (75648L*x0)));
                            tmp0.store(out_ptr2 + static_cast<long>(x3 + (64L*x2) + (12608L*x1) + (75648L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_120 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x0) % static_cast<long>(197L))) + (12608L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (75648L*(c10::div_floor_integer(x0, 197L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_121 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(384.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp13.store(out_ptr2 + static_cast<long>(x1 + (384L*x0)));
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_122 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2420736L); x0+=static_cast<long>(8L))
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


cpp_fused_clone_123 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                    {
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(6L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x3 + (6L*x1) + (48L*x2) + (768L*x0))];
                            out_ptr0[static_cast<long>(x3 + (6L*x2) + (96L*x1) + (384L*x0))] = tmp0;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(24L + x1 + (48L*x2) + (768L*x0)), static_cast<long>(48L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (16L*x1) + (16L*x1_inner) + (384L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_mul_124 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(100352L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (16L*x0)));
                        auto tmp1 = static_cast<float>(0.408248290463863);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp3);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(100352L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.408248290463863);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 - tmp5;
                        auto tmp7 = tmp6.exp();
                        tmp7.store(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp7;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(100352L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                    {
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(6L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr1[static_cast<long>(x3 + (6L*x1) + (24L*x2) + (384L*x0))];
                            out_ptr2[static_cast<long>(x3 + (6L*x2) + (96L*x1) + (384L*x0))] = tmp0;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_125 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((6L*(static_cast<long>(x0) % static_cast<long>(16L))) + (96L*(c10::div_floor_integer((x1 + x1_inner), 6L))) + (384L*(c10::div_floor_integer(x0, 16L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(6L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (24L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_126 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(602112L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (24L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(24.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (24L*x0)));
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (24L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_127 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2408448L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_view_128 = async_compile.cpp('''
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
                       float* out_ptr3,
                       float* out_ptr4)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (24L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (24L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(24.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp13.store(out_ptr2 + static_cast<long>(x1 + (24L*x0)));
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (24L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(static_cast<long>((x1 + x1_inner)) % static_cast<long>(24L))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp3 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr5[static_cast<long>(static_cast<long>((x1 + x1_inner)) % static_cast<long>(24L))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr4 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_cat_native_layer_norm_view_129 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr2)
{
    auto out_ptr1 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = c10::convert<int>(x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp4));
                            auto tmp7 = masked_load(in_ptr1 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp4));
                            auto tmp8 = tmp6 + tmp7;
                            auto tmp9 = masked_load(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp4));
                            auto tmp10 = tmp8 + tmp9;
                            return tmp10;
                        }
                        ;
                        auto tmp11 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp12 = tmp0 >= tmp3;
                        auto tmp13 = static_cast<int>(197);
                        auto tmp14 = tmp0 < tmp13;
                        auto tmp15 = [&]
                        {
                            auto tmp16 = masked_load(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp12));
                            auto tmp17 = masked_load(in_ptr1 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp12));
                            auto tmp18 = tmp16 + tmp17;
                            auto tmp19 = masked_load(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp12));
                            auto tmp20 = tmp18 + tmp19;
                            auto tmp21 = masked_load(in_ptr2 + static_cast<long>((-384L) + x2 + (384L*x1) + (75264L*x0)), to_float_mask(tmp12));
                            auto tmp22 = tmp20 + tmp21;
                            return tmp22;
                        }
                        ;
                        auto tmp23 = decltype(tmp15())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp15(), to_float_mask(tmp12));
                        auto tmp24 = to_float_mask(tmp4);
                        auto tmp25 = decltype(tmp11)::blendv(tmp23, tmp11, tmp24);
                        tmp25.store(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(384.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = in_out_ptr1[static_cast<long>(x0)];
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp10 = tmp8 + tmp9;
                    tmp10.store(out_ptr2 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_130 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (768L*x2) + (151296L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (12608L*x1) + (75648L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(384L + x1 + (768L*x2) + (151296L*x0)), static_cast<long>(768L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (197L*x1) + (197L*x1_inner) + (75648L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(384L + x1 + (768L*x2) + (151296L*x0)));
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp0.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (197L*x1) + (197L*x1_inner) + (75648L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_mul_131 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9456L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (197L*x0)));
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp3);
                    }
                    #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (197L*x0))];
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp2);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9456L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 - tmp5;
                        auto tmp7 = tmp6.exp();
                        tmp7.store(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp7;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (197L*x0))];
                        auto tmp3 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 - tmp3);
                        auto tmp5 = std::exp(tmp4);
                        in_out_ptr0[static_cast<long>(x1 + (197L*x0))] = tmp5;
                        tmp_acc0 = tmp_acc0 + tmp5;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9456L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (197L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    in_out_ptr0[static_cast<long>(x1 + (197L*x0))] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x1) + (384L*x2) + (75648L*x0)));
                            tmp0.store(out_ptr2 + static_cast<long>(x3 + (64L*x2) + (12608L*x1) + (75648L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_132 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x0) % static_cast<long>(197L))) + (12608L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (75648L*(c10::div_floor_integer(x0, 197L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_133 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(384.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp13.store(out_ptr2 + static_cast<long>(x1 + (384L*x0)));
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_134 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2420736L); x0+=static_cast<long>(8L))
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


cpp_fused_clone_135 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                    {
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(6L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x3 + (6L*x1) + (48L*x2) + (768L*x0))];
                            out_ptr0[static_cast<long>(x3 + (6L*x2) + (96L*x1) + (384L*x0))] = tmp0;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(24L + x1 + (48L*x2) + (768L*x0)), static_cast<long>(48L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (16L*x1) + (16L*x1_inner) + (384L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_mul_136 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(100352L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (16L*x0)));
                        auto tmp1 = static_cast<float>(0.408248290463863);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp3);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(100352L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.408248290463863);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 - tmp5;
                        auto tmp7 = tmp6.exp();
                        tmp7.store(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp7;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(100352L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                    {
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(6L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr1[static_cast<long>(x3 + (6L*x1) + (24L*x2) + (384L*x0))];
                            out_ptr2[static_cast<long>(x3 + (6L*x2) + (96L*x1) + (384L*x0))] = tmp0;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_137 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((6L*(static_cast<long>(x0) % static_cast<long>(16L))) + (96L*(c10::div_floor_integer((x1 + x1_inner), 6L))) + (384L*(c10::div_floor_integer(x0, 16L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(6L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (24L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_138 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (24L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (24L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (24L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(24.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-05);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (24L*x0)));
                    tmp19.store(out_ptr3 + static_cast<long>(x1 + (24L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_139 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2408448L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_view_140 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (24L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (24L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (24L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (24L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(24.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-05);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (24L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(static_cast<long>((x1 + x1_inner)) % static_cast<long>(24L))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp3 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr5[static_cast<long>(static_cast<long>((x1 + x1_inner)) % static_cast<long>(24L))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr3 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_cat_native_layer_norm_view_141 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr2)
{
    auto out_ptr1 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = c10::convert<int>(x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp4));
                            auto tmp7 = masked_load(in_ptr1 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp4));
                            auto tmp8 = tmp6 + tmp7;
                            auto tmp9 = masked_load(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp4));
                            auto tmp10 = tmp8 + tmp9;
                            return tmp10;
                        }
                        ;
                        auto tmp11 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp12 = tmp0 >= tmp3;
                        auto tmp13 = static_cast<int>(197);
                        auto tmp14 = tmp0 < tmp13;
                        auto tmp15 = [&]
                        {
                            auto tmp16 = masked_load(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp12));
                            auto tmp17 = masked_load(in_ptr1 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp12));
                            auto tmp18 = tmp16 + tmp17;
                            auto tmp19 = masked_load(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp12));
                            auto tmp20 = tmp18 + tmp19;
                            auto tmp21 = masked_load(in_ptr2 + static_cast<long>((-384L) + x2 + (384L*x1) + (75264L*x0)), to_float_mask(tmp12));
                            auto tmp22 = tmp20 + tmp21;
                            return tmp22;
                        }
                        ;
                        auto tmp23 = decltype(tmp15())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp15(), to_float_mask(tmp12));
                        auto tmp24 = to_float_mask(tmp4);
                        auto tmp25 = decltype(tmp11)::blendv(tmp23, tmp11, tmp24);
                        tmp25.store(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(384.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = in_out_ptr1[static_cast<long>(x0)];
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp10 = tmp8 + tmp9;
                    tmp10.store(out_ptr2 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_142 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (768L*x2) + (151296L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (12608L*x1) + (75648L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(384L + x1 + (768L*x2) + (151296L*x0)), static_cast<long>(768L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (197L*x1) + (197L*x1_inner) + (75648L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(384L + x1 + (768L*x2) + (151296L*x0)));
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp0.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (197L*x1) + (197L*x1_inner) + (75648L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_mul_143 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9456L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (197L*x0)));
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp3);
                    }
                    #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (197L*x0))];
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp2);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9456L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 - tmp5;
                        auto tmp7 = tmp6.exp();
                        tmp7.store(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp7;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (197L*x0))];
                        auto tmp3 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 - tmp3);
                        auto tmp5 = std::exp(tmp4);
                        in_out_ptr0[static_cast<long>(x1 + (197L*x0))] = tmp5;
                        tmp_acc0 = tmp_acc0 + tmp5;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9456L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (197L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    in_out_ptr0[static_cast<long>(x1 + (197L*x0))] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x1) + (384L*x2) + (75648L*x0)));
                            tmp0.store(out_ptr2 + static_cast<long>(x3 + (64L*x2) + (12608L*x1) + (75648L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_144 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x0) % static_cast<long>(197L))) + (12608L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (75648L*(c10::div_floor_integer(x0, 197L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_145 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(384.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp13.store(out_ptr2 + static_cast<long>(x1 + (384L*x0)));
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_146 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2420736L); x0+=static_cast<long>(8L))
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


cpp_fused_add_clone_native_layer_norm_147 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (384L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(384.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-05);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1 + (75648L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        tmp4.store(out_ptr3 + static_cast<long>(x1 + (384L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_detach_native_layer_norm_native_layer_norm_backward_148 = async_compile.cpp('''
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
                       float* out_ptr3,
                       float* out_ptr4,
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
                       float* out_ptr23)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(384.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-05);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            auto tmp8 = tmp7 / tmp2;
            tmp8.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(384.0);
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
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(38809L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (38809L*x1) + (232854L*x0))];
                        out_ptr0[static_cast<long>(x1 + (6L*x2) + (232854L*x0))] = tmp0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(24.0);
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr3 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(24.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr1[static_cast<long>(x2 + (256L*x1) + (1024L*x0))];
                        out_ptr1[static_cast<long>(x1 + (4L*x2) + (1024L*x0))] = tmp0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr4 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(24.0);
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr5 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(384.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(38809L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr2[static_cast<long>(x2 + (38809L*x1) + (232854L*x0))];
                        out_ptr2[static_cast<long>(x1 + (6L*x2) + (232854L*x0))] = tmp0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr6 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(24.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr3[static_cast<long>(x2 + (256L*x1) + (1024L*x0))];
                        out_ptr3[static_cast<long>(x1 + (4L*x2) + (1024L*x0))] = tmp0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr7 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(24.0);
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
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr8 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(384.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(38809L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr4[static_cast<long>(x2 + (38809L*x1) + (232854L*x0))];
                        out_ptr4[static_cast<long>(x1 + (6L*x2) + (232854L*x0))] = tmp0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr9 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(24.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr5[static_cast<long>(x2 + (256L*x1) + (1024L*x0))];
                        out_ptr5[static_cast<long>(x1 + (4L*x2) + (1024L*x0))] = tmp0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr10 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(24.0);
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
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr11 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(384.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(38809L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr6[static_cast<long>(x2 + (38809L*x1) + (232854L*x0))];
                        out_ptr6[static_cast<long>(x1 + (6L*x2) + (232854L*x0))] = tmp0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr12 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(24.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr7[static_cast<long>(x2 + (256L*x1) + (1024L*x0))];
                        out_ptr7[static_cast<long>(x1 + (4L*x2) + (1024L*x0))] = tmp0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr13 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(24.0);
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr14 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(384.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(38809L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr8[static_cast<long>(x2 + (38809L*x1) + (232854L*x0))];
                        out_ptr8[static_cast<long>(x1 + (6L*x2) + (232854L*x0))] = tmp0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr15 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(24.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr9[static_cast<long>(x2 + (256L*x1) + (1024L*x0))];
                        out_ptr9[static_cast<long>(x1 + (4L*x2) + (1024L*x0))] = tmp0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr16 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(24.0);
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
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr17 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(384.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(38809L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr10[static_cast<long>(x2 + (38809L*x1) + (232854L*x0))];
                        out_ptr10[static_cast<long>(x1 + (6L*x2) + (232854L*x0))] = tmp0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr18 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(24.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr11[static_cast<long>(x2 + (256L*x1) + (1024L*x0))];
                        out_ptr11[static_cast<long>(x1 + (4L*x2) + (1024L*x0))] = tmp0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr19 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(24.0);
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
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr20 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(384.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(38809L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr12[static_cast<long>(x2 + (38809L*x1) + (232854L*x0))];
                        out_ptr12[static_cast<long>(x1 + (6L*x2) + (232854L*x0))] = tmp0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr21 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(24.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr13[static_cast<long>(x2 + (256L*x1) + (1024L*x0))];
                        out_ptr13[static_cast<long>(x1 + (4L*x2) + (1024L*x0))] = tmp0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr22 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(24.0);
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
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr23 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(384.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(38809L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr14[static_cast<long>(x2 + (38809L*x1) + (232854L*x0))];
                        out_ptr14[static_cast<long>(x1 + (6L*x2) + (232854L*x0))] = tmp0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr24 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(24.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr15[static_cast<long>(x2 + (256L*x1) + (1024L*x0))];
                        out_ptr15[static_cast<long>(x1 + (4L*x2) + (1024L*x0))] = tmp0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr25 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(24.0);
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr26 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(384.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(38809L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr16[static_cast<long>(x2 + (38809L*x1) + (232854L*x0))];
                        out_ptr16[static_cast<long>(x1 + (6L*x2) + (232854L*x0))] = tmp0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr27 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(24.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr17[static_cast<long>(x2 + (256L*x1) + (1024L*x0))];
                        out_ptr17[static_cast<long>(x1 + (4L*x2) + (1024L*x0))] = tmp0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr28 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(24.0);
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
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr29 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(384.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(38809L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr18[static_cast<long>(x2 + (38809L*x1) + (232854L*x0))];
                        out_ptr18[static_cast<long>(x1 + (6L*x2) + (232854L*x0))] = tmp0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr30 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(24.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr19[static_cast<long>(x2 + (256L*x1) + (1024L*x0))];
                        out_ptr19[static_cast<long>(x1 + (4L*x2) + (1024L*x0))] = tmp0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr31 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(24.0);
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
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr32 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(384.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(38809L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr20[static_cast<long>(x2 + (38809L*x1) + (232854L*x0))];
                        out_ptr20[static_cast<long>(x1 + (6L*x2) + (232854L*x0))] = tmp0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr33 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(24.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr21[static_cast<long>(x2 + (256L*x1) + (1024L*x0))];
                        out_ptr21[static_cast<long>(x1 + (4L*x2) + (1024L*x0))] = tmp0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr34 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(24.0);
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
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr35 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(384.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(38809L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr22[static_cast<long>(x2 + (38809L*x1) + (232854L*x0))];
                        out_ptr22[static_cast<long>(x1 + (6L*x2) + (232854L*x0))] = tmp0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr36 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(24.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr23[static_cast<long>(x2 + (256L*x1) + (1024L*x0))];
                        out_ptr23[static_cast<long>(x1 + (4L*x2) + (1024L*x0))] = tmp0;
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
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352 = args
    args.clear()
    assert_size_stride(primals_1, (1, 24, 4, 4), (384, 16, 4, 1))
    assert_size_stride(primals_2, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_3, (1, 197, 384), (75648, 384, 1))
    assert_size_stride(primals_4, (24, 3, 7, 7), (147, 49, 7, 1))
    assert_size_stride(primals_5, (24, ), (1, ))
    assert_size_stride(primals_6, (384, ), (1, ))
    assert_size_stride(primals_7, (384, ), (1, ))
    assert_size_stride(primals_8, (384, 384), (384, 1))
    assert_size_stride(primals_9, (384, ), (1, ))
    assert_size_stride(primals_10, (384, ), (1, ))
    assert_size_stride(primals_11, (384, ), (1, ))
    assert_size_stride(primals_12, (24, ), (1, ))
    assert_size_stride(primals_13, (24, ), (1, ))
    assert_size_stride(primals_14, (48, 24), (24, 1))
    assert_size_stride(primals_15, (24, 24), (24, 1))
    assert_size_stride(primals_16, (24, 24), (24, 1))
    assert_size_stride(primals_17, (24, ), (1, ))
    assert_size_stride(primals_18, (24, ), (1, ))
    assert_size_stride(primals_19, (24, ), (1, ))
    assert_size_stride(primals_20, (96, 24), (24, 1))
    assert_size_stride(primals_21, (96, ), (1, ))
    assert_size_stride(primals_22, (24, 96), (96, 1))
    assert_size_stride(primals_23, (24, ), (1, ))
    assert_size_stride(primals_24, (24, ), (1, ))
    assert_size_stride(primals_25, (24, ), (1, ))
    assert_size_stride(primals_26, (384, 384), (384, 1))
    assert_size_stride(primals_27, (384, ), (1, ))
    assert_size_stride(primals_28, (384, ), (1, ))
    assert_size_stride(primals_29, (384, ), (1, ))
    assert_size_stride(primals_30, (768, 384), (384, 1))
    assert_size_stride(primals_31, (384, 384), (384, 1))
    assert_size_stride(primals_32, (384, 384), (384, 1))
    assert_size_stride(primals_33, (384, ), (1, ))
    assert_size_stride(primals_34, (384, ), (1, ))
    assert_size_stride(primals_35, (384, ), (1, ))
    assert_size_stride(primals_36, (1536, 384), (384, 1))
    assert_size_stride(primals_37, (1536, ), (1, ))
    assert_size_stride(primals_38, (384, 1536), (1536, 1))
    assert_size_stride(primals_39, (384, ), (1, ))
    assert_size_stride(primals_40, (24, ), (1, ))
    assert_size_stride(primals_41, (24, ), (1, ))
    assert_size_stride(primals_42, (48, 24), (24, 1))
    assert_size_stride(primals_43, (24, 24), (24, 1))
    assert_size_stride(primals_44, (24, 24), (24, 1))
    assert_size_stride(primals_45, (24, ), (1, ))
    assert_size_stride(primals_46, (24, ), (1, ))
    assert_size_stride(primals_47, (24, ), (1, ))
    assert_size_stride(primals_48, (96, 24), (24, 1))
    assert_size_stride(primals_49, (96, ), (1, ))
    assert_size_stride(primals_50, (24, 96), (96, 1))
    assert_size_stride(primals_51, (24, ), (1, ))
    assert_size_stride(primals_52, (24, ), (1, ))
    assert_size_stride(primals_53, (24, ), (1, ))
    assert_size_stride(primals_54, (384, 384), (384, 1))
    assert_size_stride(primals_55, (384, ), (1, ))
    assert_size_stride(primals_56, (384, ), (1, ))
    assert_size_stride(primals_57, (384, ), (1, ))
    assert_size_stride(primals_58, (768, 384), (384, 1))
    assert_size_stride(primals_59, (384, 384), (384, 1))
    assert_size_stride(primals_60, (384, 384), (384, 1))
    assert_size_stride(primals_61, (384, ), (1, ))
    assert_size_stride(primals_62, (384, ), (1, ))
    assert_size_stride(primals_63, (384, ), (1, ))
    assert_size_stride(primals_64, (1536, 384), (384, 1))
    assert_size_stride(primals_65, (1536, ), (1, ))
    assert_size_stride(primals_66, (384, 1536), (1536, 1))
    assert_size_stride(primals_67, (384, ), (1, ))
    assert_size_stride(primals_68, (24, ), (1, ))
    assert_size_stride(primals_69, (24, ), (1, ))
    assert_size_stride(primals_70, (48, 24), (24, 1))
    assert_size_stride(primals_71, (24, 24), (24, 1))
    assert_size_stride(primals_72, (24, 24), (24, 1))
    assert_size_stride(primals_73, (24, ), (1, ))
    assert_size_stride(primals_74, (24, ), (1, ))
    assert_size_stride(primals_75, (24, ), (1, ))
    assert_size_stride(primals_76, (96, 24), (24, 1))
    assert_size_stride(primals_77, (96, ), (1, ))
    assert_size_stride(primals_78, (24, 96), (96, 1))
    assert_size_stride(primals_79, (24, ), (1, ))
    assert_size_stride(primals_80, (24, ), (1, ))
    assert_size_stride(primals_81, (24, ), (1, ))
    assert_size_stride(primals_82, (384, 384), (384, 1))
    assert_size_stride(primals_83, (384, ), (1, ))
    assert_size_stride(primals_84, (384, ), (1, ))
    assert_size_stride(primals_85, (384, ), (1, ))
    assert_size_stride(primals_86, (768, 384), (384, 1))
    assert_size_stride(primals_87, (384, 384), (384, 1))
    assert_size_stride(primals_88, (384, 384), (384, 1))
    assert_size_stride(primals_89, (384, ), (1, ))
    assert_size_stride(primals_90, (384, ), (1, ))
    assert_size_stride(primals_91, (384, ), (1, ))
    assert_size_stride(primals_92, (1536, 384), (384, 1))
    assert_size_stride(primals_93, (1536, ), (1, ))
    assert_size_stride(primals_94, (384, 1536), (1536, 1))
    assert_size_stride(primals_95, (384, ), (1, ))
    assert_size_stride(primals_96, (24, ), (1, ))
    assert_size_stride(primals_97, (24, ), (1, ))
    assert_size_stride(primals_98, (48, 24), (24, 1))
    assert_size_stride(primals_99, (24, 24), (24, 1))
    assert_size_stride(primals_100, (24, 24), (24, 1))
    assert_size_stride(primals_101, (24, ), (1, ))
    assert_size_stride(primals_102, (24, ), (1, ))
    assert_size_stride(primals_103, (24, ), (1, ))
    assert_size_stride(primals_104, (96, 24), (24, 1))
    assert_size_stride(primals_105, (96, ), (1, ))
    assert_size_stride(primals_106, (24, 96), (96, 1))
    assert_size_stride(primals_107, (24, ), (1, ))
    assert_size_stride(primals_108, (24, ), (1, ))
    assert_size_stride(primals_109, (24, ), (1, ))
    assert_size_stride(primals_110, (384, 384), (384, 1))
    assert_size_stride(primals_111, (384, ), (1, ))
    assert_size_stride(primals_112, (384, ), (1, ))
    assert_size_stride(primals_113, (384, ), (1, ))
    assert_size_stride(primals_114, (768, 384), (384, 1))
    assert_size_stride(primals_115, (384, 384), (384, 1))
    assert_size_stride(primals_116, (384, 384), (384, 1))
    assert_size_stride(primals_117, (384, ), (1, ))
    assert_size_stride(primals_118, (384, ), (1, ))
    assert_size_stride(primals_119, (384, ), (1, ))
    assert_size_stride(primals_120, (1536, 384), (384, 1))
    assert_size_stride(primals_121, (1536, ), (1, ))
    assert_size_stride(primals_122, (384, 1536), (1536, 1))
    assert_size_stride(primals_123, (384, ), (1, ))
    assert_size_stride(primals_124, (24, ), (1, ))
    assert_size_stride(primals_125, (24, ), (1, ))
    assert_size_stride(primals_126, (48, 24), (24, 1))
    assert_size_stride(primals_127, (24, 24), (24, 1))
    assert_size_stride(primals_128, (24, 24), (24, 1))
    assert_size_stride(primals_129, (24, ), (1, ))
    assert_size_stride(primals_130, (24, ), (1, ))
    assert_size_stride(primals_131, (24, ), (1, ))
    assert_size_stride(primals_132, (96, 24), (24, 1))
    assert_size_stride(primals_133, (96, ), (1, ))
    assert_size_stride(primals_134, (24, 96), (96, 1))
    assert_size_stride(primals_135, (24, ), (1, ))
    assert_size_stride(primals_136, (24, ), (1, ))
    assert_size_stride(primals_137, (24, ), (1, ))
    assert_size_stride(primals_138, (384, 384), (384, 1))
    assert_size_stride(primals_139, (384, ), (1, ))
    assert_size_stride(primals_140, (384, ), (1, ))
    assert_size_stride(primals_141, (384, ), (1, ))
    assert_size_stride(primals_142, (768, 384), (384, 1))
    assert_size_stride(primals_143, (384, 384), (384, 1))
    assert_size_stride(primals_144, (384, 384), (384, 1))
    assert_size_stride(primals_145, (384, ), (1, ))
    assert_size_stride(primals_146, (384, ), (1, ))
    assert_size_stride(primals_147, (384, ), (1, ))
    assert_size_stride(primals_148, (1536, 384), (384, 1))
    assert_size_stride(primals_149, (1536, ), (1, ))
    assert_size_stride(primals_150, (384, 1536), (1536, 1))
    assert_size_stride(primals_151, (384, ), (1, ))
    assert_size_stride(primals_152, (24, ), (1, ))
    assert_size_stride(primals_153, (24, ), (1, ))
    assert_size_stride(primals_154, (48, 24), (24, 1))
    assert_size_stride(primals_155, (24, 24), (24, 1))
    assert_size_stride(primals_156, (24, 24), (24, 1))
    assert_size_stride(primals_157, (24, ), (1, ))
    assert_size_stride(primals_158, (24, ), (1, ))
    assert_size_stride(primals_159, (24, ), (1, ))
    assert_size_stride(primals_160, (96, 24), (24, 1))
    assert_size_stride(primals_161, (96, ), (1, ))
    assert_size_stride(primals_162, (24, 96), (96, 1))
    assert_size_stride(primals_163, (24, ), (1, ))
    assert_size_stride(primals_164, (24, ), (1, ))
    assert_size_stride(primals_165, (24, ), (1, ))
    assert_size_stride(primals_166, (384, 384), (384, 1))
    assert_size_stride(primals_167, (384, ), (1, ))
    assert_size_stride(primals_168, (384, ), (1, ))
    assert_size_stride(primals_169, (384, ), (1, ))
    assert_size_stride(primals_170, (768, 384), (384, 1))
    assert_size_stride(primals_171, (384, 384), (384, 1))
    assert_size_stride(primals_172, (384, 384), (384, 1))
    assert_size_stride(primals_173, (384, ), (1, ))
    assert_size_stride(primals_174, (384, ), (1, ))
    assert_size_stride(primals_175, (384, ), (1, ))
    assert_size_stride(primals_176, (1536, 384), (384, 1))
    assert_size_stride(primals_177, (1536, ), (1, ))
    assert_size_stride(primals_178, (384, 1536), (1536, 1))
    assert_size_stride(primals_179, (384, ), (1, ))
    assert_size_stride(primals_180, (24, ), (1, ))
    assert_size_stride(primals_181, (24, ), (1, ))
    assert_size_stride(primals_182, (48, 24), (24, 1))
    assert_size_stride(primals_183, (24, 24), (24, 1))
    assert_size_stride(primals_184, (24, 24), (24, 1))
    assert_size_stride(primals_185, (24, ), (1, ))
    assert_size_stride(primals_186, (24, ), (1, ))
    assert_size_stride(primals_187, (24, ), (1, ))
    assert_size_stride(primals_188, (96, 24), (24, 1))
    assert_size_stride(primals_189, (96, ), (1, ))
    assert_size_stride(primals_190, (24, 96), (96, 1))
    assert_size_stride(primals_191, (24, ), (1, ))
    assert_size_stride(primals_192, (24, ), (1, ))
    assert_size_stride(primals_193, (24, ), (1, ))
    assert_size_stride(primals_194, (384, 384), (384, 1))
    assert_size_stride(primals_195, (384, ), (1, ))
    assert_size_stride(primals_196, (384, ), (1, ))
    assert_size_stride(primals_197, (384, ), (1, ))
    assert_size_stride(primals_198, (768, 384), (384, 1))
    assert_size_stride(primals_199, (384, 384), (384, 1))
    assert_size_stride(primals_200, (384, 384), (384, 1))
    assert_size_stride(primals_201, (384, ), (1, ))
    assert_size_stride(primals_202, (384, ), (1, ))
    assert_size_stride(primals_203, (384, ), (1, ))
    assert_size_stride(primals_204, (1536, 384), (384, 1))
    assert_size_stride(primals_205, (1536, ), (1, ))
    assert_size_stride(primals_206, (384, 1536), (1536, 1))
    assert_size_stride(primals_207, (384, ), (1, ))
    assert_size_stride(primals_208, (24, ), (1, ))
    assert_size_stride(primals_209, (24, ), (1, ))
    assert_size_stride(primals_210, (48, 24), (24, 1))
    assert_size_stride(primals_211, (24, 24), (24, 1))
    assert_size_stride(primals_212, (24, 24), (24, 1))
    assert_size_stride(primals_213, (24, ), (1, ))
    assert_size_stride(primals_214, (24, ), (1, ))
    assert_size_stride(primals_215, (24, ), (1, ))
    assert_size_stride(primals_216, (96, 24), (24, 1))
    assert_size_stride(primals_217, (96, ), (1, ))
    assert_size_stride(primals_218, (24, 96), (96, 1))
    assert_size_stride(primals_219, (24, ), (1, ))
    assert_size_stride(primals_220, (24, ), (1, ))
    assert_size_stride(primals_221, (24, ), (1, ))
    assert_size_stride(primals_222, (384, 384), (384, 1))
    assert_size_stride(primals_223, (384, ), (1, ))
    assert_size_stride(primals_224, (384, ), (1, ))
    assert_size_stride(primals_225, (384, ), (1, ))
    assert_size_stride(primals_226, (768, 384), (384, 1))
    assert_size_stride(primals_227, (384, 384), (384, 1))
    assert_size_stride(primals_228, (384, 384), (384, 1))
    assert_size_stride(primals_229, (384, ), (1, ))
    assert_size_stride(primals_230, (384, ), (1, ))
    assert_size_stride(primals_231, (384, ), (1, ))
    assert_size_stride(primals_232, (1536, 384), (384, 1))
    assert_size_stride(primals_233, (1536, ), (1, ))
    assert_size_stride(primals_234, (384, 1536), (1536, 1))
    assert_size_stride(primals_235, (384, ), (1, ))
    assert_size_stride(primals_236, (24, ), (1, ))
    assert_size_stride(primals_237, (24, ), (1, ))
    assert_size_stride(primals_238, (48, 24), (24, 1))
    assert_size_stride(primals_239, (24, 24), (24, 1))
    assert_size_stride(primals_240, (24, 24), (24, 1))
    assert_size_stride(primals_241, (24, ), (1, ))
    assert_size_stride(primals_242, (24, ), (1, ))
    assert_size_stride(primals_243, (24, ), (1, ))
    assert_size_stride(primals_244, (96, 24), (24, 1))
    assert_size_stride(primals_245, (96, ), (1, ))
    assert_size_stride(primals_246, (24, 96), (96, 1))
    assert_size_stride(primals_247, (24, ), (1, ))
    assert_size_stride(primals_248, (24, ), (1, ))
    assert_size_stride(primals_249, (24, ), (1, ))
    assert_size_stride(primals_250, (384, 384), (384, 1))
    assert_size_stride(primals_251, (384, ), (1, ))
    assert_size_stride(primals_252, (384, ), (1, ))
    assert_size_stride(primals_253, (384, ), (1, ))
    assert_size_stride(primals_254, (768, 384), (384, 1))
    assert_size_stride(primals_255, (384, 384), (384, 1))
    assert_size_stride(primals_256, (384, 384), (384, 1))
    assert_size_stride(primals_257, (384, ), (1, ))
    assert_size_stride(primals_258, (384, ), (1, ))
    assert_size_stride(primals_259, (384, ), (1, ))
    assert_size_stride(primals_260, (1536, 384), (384, 1))
    assert_size_stride(primals_261, (1536, ), (1, ))
    assert_size_stride(primals_262, (384, 1536), (1536, 1))
    assert_size_stride(primals_263, (384, ), (1, ))
    assert_size_stride(primals_264, (24, ), (1, ))
    assert_size_stride(primals_265, (24, ), (1, ))
    assert_size_stride(primals_266, (48, 24), (24, 1))
    assert_size_stride(primals_267, (24, 24), (24, 1))
    assert_size_stride(primals_268, (24, 24), (24, 1))
    assert_size_stride(primals_269, (24, ), (1, ))
    assert_size_stride(primals_270, (24, ), (1, ))
    assert_size_stride(primals_271, (24, ), (1, ))
    assert_size_stride(primals_272, (96, 24), (24, 1))
    assert_size_stride(primals_273, (96, ), (1, ))
    assert_size_stride(primals_274, (24, 96), (96, 1))
    assert_size_stride(primals_275, (24, ), (1, ))
    assert_size_stride(primals_276, (24, ), (1, ))
    assert_size_stride(primals_277, (24, ), (1, ))
    assert_size_stride(primals_278, (384, 384), (384, 1))
    assert_size_stride(primals_279, (384, ), (1, ))
    assert_size_stride(primals_280, (384, ), (1, ))
    assert_size_stride(primals_281, (384, ), (1, ))
    assert_size_stride(primals_282, (768, 384), (384, 1))
    assert_size_stride(primals_283, (384, 384), (384, 1))
    assert_size_stride(primals_284, (384, 384), (384, 1))
    assert_size_stride(primals_285, (384, ), (1, ))
    assert_size_stride(primals_286, (384, ), (1, ))
    assert_size_stride(primals_287, (384, ), (1, ))
    assert_size_stride(primals_288, (1536, 384), (384, 1))
    assert_size_stride(primals_289, (1536, ), (1, ))
    assert_size_stride(primals_290, (384, 1536), (1536, 1))
    assert_size_stride(primals_291, (384, ), (1, ))
    assert_size_stride(primals_292, (24, ), (1, ))
    assert_size_stride(primals_293, (24, ), (1, ))
    assert_size_stride(primals_294, (48, 24), (24, 1))
    assert_size_stride(primals_295, (24, 24), (24, 1))
    assert_size_stride(primals_296, (24, 24), (24, 1))
    assert_size_stride(primals_297, (24, ), (1, ))
    assert_size_stride(primals_298, (24, ), (1, ))
    assert_size_stride(primals_299, (24, ), (1, ))
    assert_size_stride(primals_300, (96, 24), (24, 1))
    assert_size_stride(primals_301, (96, ), (1, ))
    assert_size_stride(primals_302, (24, 96), (96, 1))
    assert_size_stride(primals_303, (24, ), (1, ))
    assert_size_stride(primals_304, (24, ), (1, ))
    assert_size_stride(primals_305, (24, ), (1, ))
    assert_size_stride(primals_306, (384, 384), (384, 1))
    assert_size_stride(primals_307, (384, ), (1, ))
    assert_size_stride(primals_308, (384, ), (1, ))
    assert_size_stride(primals_309, (384, ), (1, ))
    assert_size_stride(primals_310, (768, 384), (384, 1))
    assert_size_stride(primals_311, (384, 384), (384, 1))
    assert_size_stride(primals_312, (384, 384), (384, 1))
    assert_size_stride(primals_313, (384, ), (1, ))
    assert_size_stride(primals_314, (384, ), (1, ))
    assert_size_stride(primals_315, (384, ), (1, ))
    assert_size_stride(primals_316, (1536, 384), (384, 1))
    assert_size_stride(primals_317, (1536, ), (1, ))
    assert_size_stride(primals_318, (384, 1536), (1536, 1))
    assert_size_stride(primals_319, (384, ), (1, ))
    assert_size_stride(primals_320, (24, ), (1, ))
    assert_size_stride(primals_321, (24, ), (1, ))
    assert_size_stride(primals_322, (48, 24), (24, 1))
    assert_size_stride(primals_323, (24, 24), (24, 1))
    assert_size_stride(primals_324, (24, 24), (24, 1))
    assert_size_stride(primals_325, (24, ), (1, ))
    assert_size_stride(primals_326, (24, ), (1, ))
    assert_size_stride(primals_327, (24, ), (1, ))
    assert_size_stride(primals_328, (96, 24), (24, 1))
    assert_size_stride(primals_329, (96, ), (1, ))
    assert_size_stride(primals_330, (24, 96), (96, 1))
    assert_size_stride(primals_331, (24, ), (1, ))
    assert_size_stride(primals_332, (24, ), (1, ))
    assert_size_stride(primals_333, (24, ), (1, ))
    assert_size_stride(primals_334, (384, 384), (384, 1))
    assert_size_stride(primals_335, (384, ), (1, ))
    assert_size_stride(primals_336, (384, ), (1, ))
    assert_size_stride(primals_337, (384, ), (1, ))
    assert_size_stride(primals_338, (768, 384), (384, 1))
    assert_size_stride(primals_339, (384, 384), (384, 1))
    assert_size_stride(primals_340, (384, 384), (384, 1))
    assert_size_stride(primals_341, (384, ), (1, ))
    assert_size_stride(primals_342, (384, ), (1, ))
    assert_size_stride(primals_343, (384, ), (1, ))
    assert_size_stride(primals_344, (1536, 384), (384, 1))
    assert_size_stride(primals_345, (1536, ), (1, ))
    assert_size_stride(primals_346, (384, 1536), (1536, 1))
    assert_size_stride(primals_347, (384, ), (1, ))
    assert_size_stride(primals_348, (384, ), (1, ))
    assert_size_stride(primals_349, (384, ), (1, ))
    assert_size_stride(primals_350, (1000, 384), (384, 1))
    assert_size_stride(primals_351, (1000, ), (1, ))
    assert_size_stride(primals_352, (8, 3, 224, 224), (150528, 50176, 224, 1))
    buf0 = empty_strided((24, 3, 7, 7), (147, 1, 21, 3), device='cpu', dtype=torch.float32)
    buf1 = empty_strided((8, 3, 224, 224), (150528, 1, 672, 3), device='cpu', dtype=torch.float32)
    cpp_fused_0(c_void_p(primals_4.data_ptr()), c_void_p(primals_352.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()))
    del primals_352
    del primals_4
    # Source Nodes: [x], Original ATen: [aten.convolution]
    buf2 = extern_kernels.convolution(buf1, buf0, primals_5, stride=(4, 4), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf2, (8, 24, 56, 56), (75264, 1, 1344, 24))
    del primals_5
    buf3 = empty((4, 14), device='cpu', dtype=torch.int64)
    buf4 = empty((1568, 16, 24), device='cpu', dtype=torch.float32)
    buf5 = empty((8, 196, 1), device='cpu', dtype=torch.float32)
    buf6 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf8 = reinterpret_tensor(buf6, (8, 196, 1), (196, 1, 1), 0); del buf6  # reuse
    buf9 = empty((1568, 384), device='cpu', dtype=torch.float32)
    cpp_fused_clone_im2col_native_layer_norm_view_1(c_void_p(buf8.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(primals_1.data_ptr()), c_void_p(primals_6.data_ptr()), c_void_p(primals_7.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(buf5.data_ptr()), c_void_p(buf9.data_ptr()))
    del primals_7
    buf10 = empty((1568, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_9, buf9, reinterpret_tensor(primals_8, (384, 384), (1, 384), 0), alpha=1, beta=1, out=buf10)
    del primals_9
    buf11 = empty((8, 196, 1), device='cpu', dtype=torch.float32)
    buf12 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf14 = reinterpret_tensor(buf12, (8, 196, 1), (196, 1, 1), 0); del buf12  # reuse
    buf15 = empty((8, 197, 384), device='cpu', dtype=torch.float32)
    buf16 = empty((1568, 16, 1), device='cpu', dtype=torch.float32)
    buf17 = empty_strided((1568, 16, 1), (16, 1, 25088), device='cpu', dtype=torch.float32)
    buf19 = reinterpret_tensor(buf17, (1568, 16, 1), (16, 1, 1), 0); del buf17  # reuse
    buf20 = empty((25088, 24), device='cpu', dtype=torch.float32)
    cpp_fused_add_cat_native_layer_norm_view_2(c_void_p(buf14.data_ptr()), c_void_p(buf19.data_ptr()), c_void_p(buf10.data_ptr()), c_void_p(primals_2.data_ptr()), c_void_p(primals_10.data_ptr()), c_void_p(primals_11.data_ptr()), c_void_p(primals_3.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(primals_12.data_ptr()), c_void_p(primals_13.data_ptr()), c_void_p(buf11.data_ptr()), c_void_p(buf15.data_ptr()), c_void_p(buf16.data_ptr()), c_void_p(buf20.data_ptr()))
    del primals_11
    del primals_13
    del primals_2
    del primals_3
    buf21 = empty((25088, 48), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___blocks_0_attn_in_qk], Original ATen: [aten.mm]
    extern_kernels.mm(buf20, reinterpret_tensor(primals_14, (24, 48), (1, 24), 0), out=buf21)
    buf22 = empty((25088, 24), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___blocks_0_attn_in_v], Original ATen: [aten.mm]
    extern_kernels.mm(buf20, reinterpret_tensor(primals_15, (24, 24), (1, 24), 0), out=buf22)
    buf23 = empty((1568, 4, 16, 6), device='cpu', dtype=torch.float32)
    buf24 = empty((1568, 4, 6, 16), device='cpu', dtype=torch.float32)
    cpp_fused_clone_3(c_void_p(buf21.data_ptr()), c_void_p(buf23.data_ptr()), c_void_p(buf24.data_ptr()))
    buf25 = empty((6272, 16, 16), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf23, (6272, 16, 6), (96, 6, 1), 0), reinterpret_tensor(buf24, (6272, 6, 16), (96, 16, 1), 0), out=buf25)
    buf26 = empty_strided((1568, 4, 16, 1), (64, 16, 1, 100352), device='cpu', dtype=torch.float32)
    buf27 = reinterpret_tensor(buf25, (1568, 4, 16, 16), (1024, 256, 16, 1), 0); del buf25  # reuse
    buf28 = empty_strided((1568, 4, 16, 1), (64, 16, 1, 100352), device='cpu', dtype=torch.float32)
    buf29 = buf27; del buf27  # reuse
    buf30 = empty((1568, 4, 16, 6), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_mul_4(c_void_p(buf29.data_ptr()), c_void_p(buf22.data_ptr()), c_void_p(buf26.data_ptr()), c_void_p(buf28.data_ptr()), c_void_p(buf30.data_ptr()))
    buf31 = reinterpret_tensor(buf22, (6272, 16, 6), (96, 6, 1), 0); del buf22  # reuse
    # Source Nodes: [matmul_1], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf29, (6272, 16, 16), (256, 16, 1), 0), reinterpret_tensor(buf30, (6272, 16, 6), (96, 6, 1), 0), out=buf31)
    buf32 = empty((25088, 24), device='cpu', dtype=torch.float32)
    cpp_fused_view_5(c_void_p(buf31.data_ptr()), c_void_p(buf32.data_ptr()))
    buf33 = reinterpret_tensor(buf31, (25088, 24), (24, 1), 0); del buf31  # reuse
    # Source Nodes: [x_6], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_17, buf32, reinterpret_tensor(primals_16, (24, 24), (1, 24), 0), alpha=1, beta=1, out=buf33)
    del primals_17
    buf34 = reinterpret_tensor(buf33, (1568, 16, 24), (384, 24, 1), 0); del buf33  # reuse
    buf35 = empty_strided((1568, 16, 1), (16, 1, 25088), device='cpu', dtype=torch.float32)
    buf36 = empty_strided((1568, 16, 1), (16, 1, 25088), device='cpu', dtype=torch.float32)
    buf38 = empty((1568, 16, 24), device='cpu', dtype=torch.float32)
    buf39 = empty((25088, 24), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_6(c_void_p(buf34.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(primals_1.data_ptr()), c_void_p(primals_18.data_ptr()), c_void_p(primals_19.data_ptr()), c_void_p(buf35.data_ptr()), c_void_p(buf36.data_ptr()), c_void_p(buf38.data_ptr()), c_void_p(buf39.data_ptr()))
    del primals_1
    del primals_19
    buf40 = empty((25088, 96), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_8], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_21, buf39, reinterpret_tensor(primals_20, (24, 96), (1, 24), 0), alpha=1, beta=1, out=buf40)
    del primals_21
    buf41 = empty((25088, 96), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_7(c_void_p(buf40.data_ptr()), c_void_p(buf41.data_ptr()))
    buf42 = reinterpret_tensor(buf2, (25088, 24), (24, 1), 0); del buf2  # reuse
    # Source Nodes: [x_12], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_23, buf41, reinterpret_tensor(primals_22, (96, 24), (1, 96), 0), alpha=1, beta=1, out=buf42)
    del primals_23
    buf43 = buf35; del buf35  # reuse
    buf44 = empty_strided((1568, 16, 1), (16, 1, 25088), device='cpu', dtype=torch.float32)
    buf46 = empty((1568, 16, 24), device='cpu', dtype=torch.float32)
    buf76 = empty((25088, 24), device='cpu', dtype=torch.float32)
    buf47 = empty((1568, 384), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_8(c_void_p(buf34.data_ptr()), c_void_p(buf42.data_ptr()), c_void_p(primals_40.data_ptr()), c_void_p(primals_41.data_ptr()), c_void_p(primals_24.data_ptr()), c_void_p(primals_25.data_ptr()), c_void_p(buf43.data_ptr()), c_void_p(buf44.data_ptr()), c_void_p(buf46.data_ptr()), c_void_p(buf76.data_ptr()), c_void_p(buf47.data_ptr()))
    del primals_25
    del primals_41
    buf48 = empty((1568, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___blocks_0_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_27, buf47, reinterpret_tensor(primals_26, (384, 384), (1, 384), 0), alpha=1, beta=1, out=buf48)
    del primals_27
    buf49 = empty((8, 197, 384), device='cpu', dtype=torch.float32)
    buf50 = empty((8, 197, 1), device='cpu', dtype=torch.float32)
    buf51 = empty_strided((8, 197, 1), (197, 1, 1576), device='cpu', dtype=torch.float32)
    buf53 = reinterpret_tensor(buf51, (8, 197, 1), (197, 1, 1), 0); del buf51  # reuse
    buf54 = empty((1576, 384), device='cpu', dtype=torch.float32)
    cpp_fused_cat_native_layer_norm_view_9(c_void_p(buf53.data_ptr()), c_void_p(buf15.data_ptr()), c_void_p(buf48.data_ptr()), c_void_p(primals_28.data_ptr()), c_void_p(primals_29.data_ptr()), c_void_p(buf49.data_ptr()), c_void_p(buf50.data_ptr()), c_void_p(buf54.data_ptr()))
    del primals_29
    buf55 = empty((1576, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___blocks_0_attn_out_qk], Original ATen: [aten.mm]
    extern_kernels.mm(buf54, reinterpret_tensor(primals_30, (384, 768), (1, 384), 0), out=buf55)
    buf56 = reinterpret_tensor(buf15, (1576, 384), (384, 1), 0); del buf15  # reuse
    # Source Nodes: [l__mod___blocks_0_attn_out_v], Original ATen: [aten.mm]
    extern_kernels.mm(buf54, reinterpret_tensor(primals_31, (384, 384), (1, 384), 0), out=buf56)
    buf57 = empty((8, 6, 197, 64), device='cpu', dtype=torch.float32)
    buf58 = empty((8, 6, 64, 197), device='cpu', dtype=torch.float32)
    cpp_fused_clone_10(c_void_p(buf55.data_ptr()), c_void_p(buf57.data_ptr()), c_void_p(buf58.data_ptr()))
    buf59 = empty((48, 197, 197), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_2], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf57, (48, 197, 64), (12608, 64, 1), 0), reinterpret_tensor(buf58, (48, 64, 197), (12608, 197, 1), 0), out=buf59)
    buf60 = empty_strided((8, 6, 197, 1), (1182, 197, 1, 9456), device='cpu', dtype=torch.float32)
    buf61 = reinterpret_tensor(buf59, (8, 6, 197, 197), (232854, 38809, 197, 1), 0); del buf59  # reuse
    buf62 = empty_strided((8, 6, 197, 1), (1182, 197, 1, 9456), device='cpu', dtype=torch.float32)
    buf63 = buf61; del buf61  # reuse
    buf64 = empty((8, 6, 197, 64), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_mul_11(c_void_p(buf63.data_ptr()), c_void_p(buf56.data_ptr()), c_void_p(buf60.data_ptr()), c_void_p(buf62.data_ptr()), c_void_p(buf64.data_ptr()))
    buf65 = reinterpret_tensor(buf56, (48, 197, 64), (12608, 64, 1), 0); del buf56  # reuse
    # Source Nodes: [matmul_3], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf63, (48, 197, 197), (38809, 197, 1), 0), reinterpret_tensor(buf64, (48, 197, 64), (12608, 64, 1), 0), out=buf65)
    buf66 = empty((1576, 384), device='cpu', dtype=torch.float32)
    cpp_fused_view_12(c_void_p(buf65.data_ptr()), c_void_p(buf66.data_ptr()))
    buf67 = reinterpret_tensor(buf65, (1576, 384), (384, 1), 0); del buf65  # reuse
    # Source Nodes: [x_15], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_33, buf66, reinterpret_tensor(primals_32, (384, 384), (1, 384), 0), alpha=1, beta=1, out=buf67)
    del primals_33
    buf68 = empty_strided((8, 197, 1), (197, 1, 1576), device='cpu', dtype=torch.float32)
    buf69 = empty_strided((8, 197, 1), (197, 1, 1576), device='cpu', dtype=torch.float32)
    buf71 = empty((8, 197, 384), device='cpu', dtype=torch.float32)
    buf72 = empty((1576, 384), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_13(c_void_p(buf49.data_ptr()), c_void_p(buf67.data_ptr()), c_void_p(primals_34.data_ptr()), c_void_p(primals_35.data_ptr()), c_void_p(buf68.data_ptr()), c_void_p(buf69.data_ptr()), c_void_p(buf71.data_ptr()), c_void_p(buf72.data_ptr()))
    del primals_35
    buf73 = empty((1576, 1536), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_17], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_37, buf72, reinterpret_tensor(primals_36, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf73)
    del primals_37
    buf74 = empty((1576, 1536), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_14(c_void_p(buf73.data_ptr()), c_void_p(buf74.data_ptr()))
    buf75 = empty((1576, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_21], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_39, buf74, reinterpret_tensor(primals_38, (1536, 384), (1, 1536), 0), alpha=1, beta=1, out=buf75)
    del primals_39
    buf77 = buf21; del buf21  # reuse
    # Source Nodes: [l__mod___blocks_1_attn_in_qk], Original ATen: [aten.mm]
    extern_kernels.mm(buf76, reinterpret_tensor(primals_42, (24, 48), (1, 24), 0), out=buf77)
    buf78 = reinterpret_tensor(buf48, (25088, 24), (24, 1), 0); del buf48  # reuse
    # Source Nodes: [l__mod___blocks_1_attn_in_v], Original ATen: [aten.mm]
    extern_kernels.mm(buf76, reinterpret_tensor(primals_43, (24, 24), (1, 24), 0), out=buf78)
    buf79 = empty((1568, 4, 16, 6), device='cpu', dtype=torch.float32)
    buf80 = empty((1568, 4, 6, 16), device='cpu', dtype=torch.float32)
    cpp_fused_clone_15(c_void_p(buf77.data_ptr()), c_void_p(buf79.data_ptr()), c_void_p(buf80.data_ptr()))
    buf81 = empty((6272, 16, 16), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_4], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf79, (6272, 16, 6), (96, 6, 1), 0), reinterpret_tensor(buf80, (6272, 6, 16), (96, 16, 1), 0), out=buf81)
    buf82 = buf28; del buf28  # reuse
    buf83 = reinterpret_tensor(buf81, (1568, 4, 16, 16), (1024, 256, 16, 1), 0); del buf81  # reuse
    buf84 = buf26; del buf26  # reuse
    buf85 = buf83; del buf83  # reuse
    buf86 = empty((1568, 4, 16, 6), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_mul_16(c_void_p(buf85.data_ptr()), c_void_p(buf78.data_ptr()), c_void_p(buf82.data_ptr()), c_void_p(buf84.data_ptr()), c_void_p(buf86.data_ptr()))
    buf87 = reinterpret_tensor(buf78, (6272, 16, 6), (96, 6, 1), 0); del buf78  # reuse
    # Source Nodes: [matmul_5], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf85, (6272, 16, 16), (256, 16, 1), 0), reinterpret_tensor(buf86, (6272, 16, 6), (96, 6, 1), 0), out=buf87)
    buf88 = empty((25088, 24), device='cpu', dtype=torch.float32)
    cpp_fused_view_17(c_void_p(buf87.data_ptr()), c_void_p(buf88.data_ptr()))
    buf89 = reinterpret_tensor(buf87, (25088, 24), (24, 1), 0); del buf87  # reuse
    # Source Nodes: [x_24], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_45, buf88, reinterpret_tensor(primals_44, (24, 24), (1, 24), 0), alpha=1, beta=1, out=buf89)
    del primals_45
    buf90 = buf43; del buf43  # reuse
    buf91 = empty_strided((1568, 16, 1), (16, 1, 25088), device='cpu', dtype=torch.float32)
    buf93 = empty((1568, 16, 24), device='cpu', dtype=torch.float32)
    buf94 = empty((25088, 24), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_18(c_void_p(buf34.data_ptr()), c_void_p(buf42.data_ptr()), c_void_p(buf89.data_ptr()), c_void_p(primals_46.data_ptr()), c_void_p(primals_47.data_ptr()), c_void_p(buf90.data_ptr()), c_void_p(buf91.data_ptr()), c_void_p(buf93.data_ptr()), c_void_p(buf94.data_ptr()))
    del primals_47
    buf95 = empty((25088, 96), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_26], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_49, buf94, reinterpret_tensor(primals_48, (24, 96), (1, 24), 0), alpha=1, beta=1, out=buf95)
    del primals_49
    buf96 = empty((25088, 96), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_19(c_void_p(buf95.data_ptr()), c_void_p(buf96.data_ptr()))
    buf97 = empty((25088, 24), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_30], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_51, buf96, reinterpret_tensor(primals_50, (96, 24), (1, 96), 0), alpha=1, beta=1, out=buf97)
    del primals_51
    buf98 = buf90; del buf90  # reuse
    buf99 = empty_strided((1568, 16, 1), (16, 1, 25088), device='cpu', dtype=torch.float32)
    buf101 = empty((1568, 16, 24), device='cpu', dtype=torch.float32)
    buf131 = empty((25088, 24), device='cpu', dtype=torch.float32)
    buf102 = empty((1568, 384), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_20(c_void_p(buf34.data_ptr()), c_void_p(buf42.data_ptr()), c_void_p(buf89.data_ptr()), c_void_p(buf97.data_ptr()), c_void_p(primals_68.data_ptr()), c_void_p(primals_69.data_ptr()), c_void_p(primals_52.data_ptr()), c_void_p(primals_53.data_ptr()), c_void_p(buf98.data_ptr()), c_void_p(buf99.data_ptr()), c_void_p(buf101.data_ptr()), c_void_p(buf131.data_ptr()), c_void_p(buf102.data_ptr()))
    del primals_53
    del primals_69
    buf103 = empty((1568, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___blocks_1_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_55, buf102, reinterpret_tensor(primals_54, (384, 384), (1, 384), 0), alpha=1, beta=1, out=buf103)
    del primals_55
    buf104 = reinterpret_tensor(buf75, (8, 197, 384), (75648, 384, 1), 0); del buf75  # reuse
    buf105 = reinterpret_tensor(buf68, (8, 197, 1), (197, 1, 1), 0); del buf68  # reuse
    buf106 = empty_strided((8, 197, 1), (197, 1, 1576), device='cpu', dtype=torch.float32)
    buf108 = reinterpret_tensor(buf106, (8, 197, 1), (197, 1, 1), 0); del buf106  # reuse
    buf109 = empty((1576, 384), device='cpu', dtype=torch.float32)
    cpp_fused_cat_native_layer_norm_view_21(c_void_p(buf104.data_ptr()), c_void_p(buf108.data_ptr()), c_void_p(buf49.data_ptr()), c_void_p(buf67.data_ptr()), c_void_p(buf103.data_ptr()), c_void_p(primals_56.data_ptr()), c_void_p(primals_57.data_ptr()), c_void_p(buf105.data_ptr()), c_void_p(buf109.data_ptr()))
    del primals_57
    buf110 = buf55; del buf55  # reuse
    # Source Nodes: [l__mod___blocks_1_attn_out_qk], Original ATen: [aten.mm]
    extern_kernels.mm(buf109, reinterpret_tensor(primals_58, (384, 768), (1, 384), 0), out=buf110)
    buf111 = buf67; del buf67  # reuse
    # Source Nodes: [l__mod___blocks_1_attn_out_v], Original ATen: [aten.mm]
    extern_kernels.mm(buf109, reinterpret_tensor(primals_59, (384, 384), (1, 384), 0), out=buf111)
    buf112 = empty((8, 6, 197, 64), device='cpu', dtype=torch.float32)
    buf113 = empty((8, 6, 64, 197), device='cpu', dtype=torch.float32)
    cpp_fused_clone_22(c_void_p(buf110.data_ptr()), c_void_p(buf112.data_ptr()), c_void_p(buf113.data_ptr()))
    buf114 = empty((48, 197, 197), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_6], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf112, (48, 197, 64), (12608, 64, 1), 0), reinterpret_tensor(buf113, (48, 64, 197), (12608, 197, 1), 0), out=buf114)
    buf115 = buf62; del buf62  # reuse
    buf116 = reinterpret_tensor(buf114, (8, 6, 197, 197), (232854, 38809, 197, 1), 0); del buf114  # reuse
    buf117 = buf60; del buf60  # reuse
    buf118 = buf116; del buf116  # reuse
    buf119 = empty((8, 6, 197, 64), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_mul_23(c_void_p(buf118.data_ptr()), c_void_p(buf111.data_ptr()), c_void_p(buf115.data_ptr()), c_void_p(buf117.data_ptr()), c_void_p(buf119.data_ptr()))
    buf120 = reinterpret_tensor(buf111, (48, 197, 64), (12608, 64, 1), 0); del buf111  # reuse
    # Source Nodes: [matmul_7], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf118, (48, 197, 197), (38809, 197, 1), 0), reinterpret_tensor(buf119, (48, 197, 64), (12608, 64, 1), 0), out=buf120)
    buf121 = empty((1576, 384), device='cpu', dtype=torch.float32)
    cpp_fused_view_24(c_void_p(buf120.data_ptr()), c_void_p(buf121.data_ptr()))
    buf122 = reinterpret_tensor(buf120, (1576, 384), (384, 1), 0); del buf120  # reuse
    # Source Nodes: [x_33], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_61, buf121, reinterpret_tensor(primals_60, (384, 384), (1, 384), 0), alpha=1, beta=1, out=buf122)
    del primals_61
    buf123 = empty_strided((8, 197, 1), (197, 1, 1576), device='cpu', dtype=torch.float32)
    buf124 = empty_strided((8, 197, 1), (197, 1, 1576), device='cpu', dtype=torch.float32)
    buf126 = empty((8, 197, 384), device='cpu', dtype=torch.float32)
    buf127 = empty((1576, 384), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_25(c_void_p(buf104.data_ptr()), c_void_p(buf122.data_ptr()), c_void_p(primals_62.data_ptr()), c_void_p(primals_63.data_ptr()), c_void_p(buf123.data_ptr()), c_void_p(buf124.data_ptr()), c_void_p(buf126.data_ptr()), c_void_p(buf127.data_ptr()))
    del primals_63
    buf128 = empty((1576, 1536), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_35], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_65, buf127, reinterpret_tensor(primals_64, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf128)
    del primals_65
    buf129 = empty((1576, 1536), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_26(c_void_p(buf128.data_ptr()), c_void_p(buf129.data_ptr()))
    buf130 = empty((1576, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_39], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_67, buf129, reinterpret_tensor(primals_66, (1536, 384), (1, 1536), 0), alpha=1, beta=1, out=buf130)
    del primals_67
    buf132 = buf77; del buf77  # reuse
    # Source Nodes: [l__mod___blocks_2_attn_in_qk], Original ATen: [aten.mm]
    extern_kernels.mm(buf131, reinterpret_tensor(primals_70, (24, 48), (1, 24), 0), out=buf132)
    buf133 = reinterpret_tensor(buf103, (25088, 24), (24, 1), 0); del buf103  # reuse
    # Source Nodes: [l__mod___blocks_2_attn_in_v], Original ATen: [aten.mm]
    extern_kernels.mm(buf131, reinterpret_tensor(primals_71, (24, 24), (1, 24), 0), out=buf133)
    buf134 = empty((1568, 4, 16, 6), device='cpu', dtype=torch.float32)
    buf135 = empty((1568, 4, 6, 16), device='cpu', dtype=torch.float32)
    cpp_fused_clone_27(c_void_p(buf132.data_ptr()), c_void_p(buf134.data_ptr()), c_void_p(buf135.data_ptr()))
    buf136 = empty((6272, 16, 16), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_8], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf134, (6272, 16, 6), (96, 6, 1), 0), reinterpret_tensor(buf135, (6272, 6, 16), (96, 16, 1), 0), out=buf136)
    buf137 = buf84; del buf84  # reuse
    buf138 = reinterpret_tensor(buf136, (1568, 4, 16, 16), (1024, 256, 16, 1), 0); del buf136  # reuse
    buf139 = buf82; del buf82  # reuse
    buf140 = buf138; del buf138  # reuse
    buf141 = empty((1568, 4, 16, 6), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_mul_28(c_void_p(buf140.data_ptr()), c_void_p(buf133.data_ptr()), c_void_p(buf137.data_ptr()), c_void_p(buf139.data_ptr()), c_void_p(buf141.data_ptr()))
    buf142 = reinterpret_tensor(buf133, (6272, 16, 6), (96, 6, 1), 0); del buf133  # reuse
    # Source Nodes: [matmul_9], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf140, (6272, 16, 16), (256, 16, 1), 0), reinterpret_tensor(buf141, (6272, 16, 6), (96, 6, 1), 0), out=buf142)
    buf143 = empty((25088, 24), device='cpu', dtype=torch.float32)
    cpp_fused_view_29(c_void_p(buf142.data_ptr()), c_void_p(buf143.data_ptr()))
    buf144 = reinterpret_tensor(buf142, (25088, 24), (24, 1), 0); del buf142  # reuse
    # Source Nodes: [x_42], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_73, buf143, reinterpret_tensor(primals_72, (24, 24), (1, 24), 0), alpha=1, beta=1, out=buf144)
    del primals_73
    buf145 = reinterpret_tensor(buf144, (1568, 16, 24), (384, 24, 1), 0); del buf144  # reuse
    buf146 = buf98; del buf98  # reuse
    buf147 = empty_strided((1568, 16, 1), (16, 1, 25088), device='cpu', dtype=torch.float32)
    buf149 = empty((1568, 16, 24), device='cpu', dtype=torch.float32)
    buf150 = empty((25088, 24), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_30(c_void_p(buf145.data_ptr()), c_void_p(buf34.data_ptr()), c_void_p(buf42.data_ptr()), c_void_p(buf89.data_ptr()), c_void_p(buf97.data_ptr()), c_void_p(primals_74.data_ptr()), c_void_p(primals_75.data_ptr()), c_void_p(buf146.data_ptr()), c_void_p(buf147.data_ptr()), c_void_p(buf149.data_ptr()), c_void_p(buf150.data_ptr()))
    del primals_75
    buf151 = empty((25088, 96), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_44], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_77, buf150, reinterpret_tensor(primals_76, (24, 96), (1, 24), 0), alpha=1, beta=1, out=buf151)
    del primals_77
    buf152 = empty((25088, 96), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_31(c_void_p(buf151.data_ptr()), c_void_p(buf152.data_ptr()))
    buf153 = buf97; del buf97  # reuse
    # Source Nodes: [x_48], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_79, buf152, reinterpret_tensor(primals_78, (96, 24), (1, 96), 0), alpha=1, beta=1, out=buf153)
    del primals_79
    buf154 = buf146; del buf146  # reuse
    buf155 = empty_strided((1568, 16, 1), (16, 1, 25088), device='cpu', dtype=torch.float32)
    buf157 = reinterpret_tensor(buf89, (1568, 16, 24), (384, 24, 1), 0); del buf89  # reuse
    buf187 = buf42; del buf42  # reuse
    buf158 = reinterpret_tensor(buf34, (1568, 384), (384, 1), 0); del buf34  # reuse
    cpp_fused_add_native_layer_norm_view_32(c_void_p(buf145.data_ptr()), c_void_p(buf153.data_ptr()), c_void_p(primals_96.data_ptr()), c_void_p(primals_97.data_ptr()), c_void_p(primals_80.data_ptr()), c_void_p(primals_81.data_ptr()), c_void_p(buf154.data_ptr()), c_void_p(buf155.data_ptr()), c_void_p(buf157.data_ptr()), c_void_p(buf187.data_ptr()), c_void_p(buf158.data_ptr()))
    del primals_81
    del primals_97
    buf159 = empty((1568, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___blocks_2_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_83, buf158, reinterpret_tensor(primals_82, (384, 384), (1, 384), 0), alpha=1, beta=1, out=buf159)
    del primals_83
    buf160 = reinterpret_tensor(buf130, (8, 197, 384), (75648, 384, 1), 0); del buf130  # reuse
    buf161 = reinterpret_tensor(buf123, (8, 197, 1), (197, 1, 1), 0); del buf123  # reuse
    buf162 = empty_strided((8, 197, 1), (197, 1, 1576), device='cpu', dtype=torch.float32)
    buf164 = reinterpret_tensor(buf162, (8, 197, 1), (197, 1, 1), 0); del buf162  # reuse
    buf165 = empty((1576, 384), device='cpu', dtype=torch.float32)
    cpp_fused_cat_native_layer_norm_view_33(c_void_p(buf160.data_ptr()), c_void_p(buf164.data_ptr()), c_void_p(buf104.data_ptr()), c_void_p(buf122.data_ptr()), c_void_p(buf159.data_ptr()), c_void_p(primals_84.data_ptr()), c_void_p(primals_85.data_ptr()), c_void_p(buf161.data_ptr()), c_void_p(buf165.data_ptr()))
    del primals_85
    buf166 = buf110; del buf110  # reuse
    # Source Nodes: [l__mod___blocks_2_attn_out_qk], Original ATen: [aten.mm]
    extern_kernels.mm(buf165, reinterpret_tensor(primals_86, (384, 768), (1, 384), 0), out=buf166)
    buf167 = buf122; del buf122  # reuse
    # Source Nodes: [l__mod___blocks_2_attn_out_v], Original ATen: [aten.mm]
    extern_kernels.mm(buf165, reinterpret_tensor(primals_87, (384, 384), (1, 384), 0), out=buf167)
    buf168 = empty((8, 6, 197, 64), device='cpu', dtype=torch.float32)
    buf169 = empty((8, 6, 64, 197), device='cpu', dtype=torch.float32)
    cpp_fused_clone_34(c_void_p(buf166.data_ptr()), c_void_p(buf168.data_ptr()), c_void_p(buf169.data_ptr()))
    buf170 = empty((48, 197, 197), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_10], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf168, (48, 197, 64), (12608, 64, 1), 0), reinterpret_tensor(buf169, (48, 64, 197), (12608, 197, 1), 0), out=buf170)
    buf171 = buf117; del buf117  # reuse
    buf172 = reinterpret_tensor(buf170, (8, 6, 197, 197), (232854, 38809, 197, 1), 0); del buf170  # reuse
    buf173 = buf115; del buf115  # reuse
    buf174 = buf172; del buf172  # reuse
    buf175 = empty((8, 6, 197, 64), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_mul_35(c_void_p(buf174.data_ptr()), c_void_p(buf167.data_ptr()), c_void_p(buf171.data_ptr()), c_void_p(buf173.data_ptr()), c_void_p(buf175.data_ptr()))
    buf176 = reinterpret_tensor(buf167, (48, 197, 64), (12608, 64, 1), 0); del buf167  # reuse
    # Source Nodes: [matmul_11], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf174, (48, 197, 197), (38809, 197, 1), 0), reinterpret_tensor(buf175, (48, 197, 64), (12608, 64, 1), 0), out=buf176)
    buf177 = empty((1576, 384), device='cpu', dtype=torch.float32)
    cpp_fused_view_36(c_void_p(buf176.data_ptr()), c_void_p(buf177.data_ptr()))
    buf178 = reinterpret_tensor(buf176, (1576, 384), (384, 1), 0); del buf176  # reuse
    # Source Nodes: [x_51], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_89, buf177, reinterpret_tensor(primals_88, (384, 384), (1, 384), 0), alpha=1, beta=1, out=buf178)
    del primals_89
    buf179 = empty_strided((8, 197, 1), (197, 1, 1576), device='cpu', dtype=torch.float32)
    buf180 = empty_strided((8, 197, 1), (197, 1, 1576), device='cpu', dtype=torch.float32)
    buf182 = empty((8, 197, 384), device='cpu', dtype=torch.float32)
    buf183 = empty((1576, 384), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_37(c_void_p(buf160.data_ptr()), c_void_p(buf178.data_ptr()), c_void_p(primals_90.data_ptr()), c_void_p(primals_91.data_ptr()), c_void_p(buf179.data_ptr()), c_void_p(buf180.data_ptr()), c_void_p(buf182.data_ptr()), c_void_p(buf183.data_ptr()))
    del primals_91
    buf184 = empty((1576, 1536), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_53], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_93, buf183, reinterpret_tensor(primals_92, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf184)
    del primals_93
    buf185 = empty((1576, 1536), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_38(c_void_p(buf184.data_ptr()), c_void_p(buf185.data_ptr()))
    buf186 = empty((1576, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_57], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_95, buf185, reinterpret_tensor(primals_94, (1536, 384), (1, 1536), 0), alpha=1, beta=1, out=buf186)
    del primals_95
    buf188 = buf132; del buf132  # reuse
    # Source Nodes: [l__mod___blocks_3_attn_in_qk], Original ATen: [aten.mm]
    extern_kernels.mm(buf187, reinterpret_tensor(primals_98, (24, 48), (1, 24), 0), out=buf188)
    buf189 = reinterpret_tensor(buf159, (25088, 24), (24, 1), 0); del buf159  # reuse
    # Source Nodes: [l__mod___blocks_3_attn_in_v], Original ATen: [aten.mm]
    extern_kernels.mm(buf187, reinterpret_tensor(primals_99, (24, 24), (1, 24), 0), out=buf189)
    buf190 = empty((1568, 4, 16, 6), device='cpu', dtype=torch.float32)
    buf191 = empty((1568, 4, 6, 16), device='cpu', dtype=torch.float32)
    cpp_fused_clone_39(c_void_p(buf188.data_ptr()), c_void_p(buf190.data_ptr()), c_void_p(buf191.data_ptr()))
    buf192 = empty((6272, 16, 16), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_12], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf190, (6272, 16, 6), (96, 6, 1), 0), reinterpret_tensor(buf191, (6272, 6, 16), (96, 16, 1), 0), out=buf192)
    buf193 = buf139; del buf139  # reuse
    buf194 = reinterpret_tensor(buf192, (1568, 4, 16, 16), (1024, 256, 16, 1), 0); del buf192  # reuse
    buf195 = buf137; del buf137  # reuse
    buf196 = buf194; del buf194  # reuse
    buf197 = empty((1568, 4, 16, 6), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_mul_40(c_void_p(buf196.data_ptr()), c_void_p(buf189.data_ptr()), c_void_p(buf193.data_ptr()), c_void_p(buf195.data_ptr()), c_void_p(buf197.data_ptr()))
    buf198 = reinterpret_tensor(buf189, (6272, 16, 6), (96, 6, 1), 0); del buf189  # reuse
    # Source Nodes: [matmul_13], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf196, (6272, 16, 16), (256, 16, 1), 0), reinterpret_tensor(buf197, (6272, 16, 6), (96, 6, 1), 0), out=buf198)
    buf199 = empty((25088, 24), device='cpu', dtype=torch.float32)
    cpp_fused_view_41(c_void_p(buf198.data_ptr()), c_void_p(buf199.data_ptr()))
    buf200 = reinterpret_tensor(buf198, (25088, 24), (24, 1), 0); del buf198  # reuse
    # Source Nodes: [x_60], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_101, buf199, reinterpret_tensor(primals_100, (24, 24), (1, 24), 0), alpha=1, beta=1, out=buf200)
    del primals_101
    buf201 = buf154; del buf154  # reuse
    buf202 = empty_strided((1568, 16, 1), (16, 1, 25088), device='cpu', dtype=torch.float32)
    buf204 = empty((1568, 16, 24), device='cpu', dtype=torch.float32)
    buf205 = empty((25088, 24), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_42(c_void_p(buf145.data_ptr()), c_void_p(buf153.data_ptr()), c_void_p(buf200.data_ptr()), c_void_p(primals_102.data_ptr()), c_void_p(primals_103.data_ptr()), c_void_p(buf201.data_ptr()), c_void_p(buf202.data_ptr()), c_void_p(buf204.data_ptr()), c_void_p(buf205.data_ptr()))
    del primals_103
    buf206 = empty((25088, 96), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_62], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_105, buf205, reinterpret_tensor(primals_104, (24, 96), (1, 24), 0), alpha=1, beta=1, out=buf206)
    del primals_105
    buf207 = empty((25088, 96), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_43(c_void_p(buf206.data_ptr()), c_void_p(buf207.data_ptr()))
    buf208 = empty((25088, 24), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_66], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_107, buf207, reinterpret_tensor(primals_106, (96, 24), (1, 96), 0), alpha=1, beta=1, out=buf208)
    del primals_107
    buf209 = buf201; del buf201  # reuse
    buf210 = empty_strided((1568, 16, 1), (16, 1, 25088), device='cpu', dtype=torch.float32)
    buf212 = empty((1568, 16, 24), device='cpu', dtype=torch.float32)
    buf242 = empty((25088, 24), device='cpu', dtype=torch.float32)
    buf213 = empty((1568, 384), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_44(c_void_p(buf145.data_ptr()), c_void_p(buf153.data_ptr()), c_void_p(buf200.data_ptr()), c_void_p(buf208.data_ptr()), c_void_p(primals_124.data_ptr()), c_void_p(primals_125.data_ptr()), c_void_p(primals_108.data_ptr()), c_void_p(primals_109.data_ptr()), c_void_p(buf209.data_ptr()), c_void_p(buf210.data_ptr()), c_void_p(buf212.data_ptr()), c_void_p(buf242.data_ptr()), c_void_p(buf213.data_ptr()))
    del primals_109
    del primals_125
    buf214 = empty((1568, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___blocks_3_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_111, buf213, reinterpret_tensor(primals_110, (384, 384), (1, 384), 0), alpha=1, beta=1, out=buf214)
    del primals_111
    buf215 = reinterpret_tensor(buf186, (8, 197, 384), (75648, 384, 1), 0); del buf186  # reuse
    buf216 = reinterpret_tensor(buf179, (8, 197, 1), (197, 1, 1), 0); del buf179  # reuse
    buf217 = empty_strided((8, 197, 1), (197, 1, 1576), device='cpu', dtype=torch.float32)
    buf219 = reinterpret_tensor(buf217, (8, 197, 1), (197, 1, 1), 0); del buf217  # reuse
    buf220 = empty((1576, 384), device='cpu', dtype=torch.float32)
    cpp_fused_cat_native_layer_norm_view_45(c_void_p(buf215.data_ptr()), c_void_p(buf219.data_ptr()), c_void_p(buf160.data_ptr()), c_void_p(buf178.data_ptr()), c_void_p(buf214.data_ptr()), c_void_p(primals_112.data_ptr()), c_void_p(primals_113.data_ptr()), c_void_p(buf216.data_ptr()), c_void_p(buf220.data_ptr()))
    del primals_113
    buf221 = buf166; del buf166  # reuse
    # Source Nodes: [l__mod___blocks_3_attn_out_qk], Original ATen: [aten.mm]
    extern_kernels.mm(buf220, reinterpret_tensor(primals_114, (384, 768), (1, 384), 0), out=buf221)
    buf222 = buf178; del buf178  # reuse
    # Source Nodes: [l__mod___blocks_3_attn_out_v], Original ATen: [aten.mm]
    extern_kernels.mm(buf220, reinterpret_tensor(primals_115, (384, 384), (1, 384), 0), out=buf222)
    buf223 = empty((8, 6, 197, 64), device='cpu', dtype=torch.float32)
    buf224 = empty((8, 6, 64, 197), device='cpu', dtype=torch.float32)
    cpp_fused_clone_46(c_void_p(buf221.data_ptr()), c_void_p(buf223.data_ptr()), c_void_p(buf224.data_ptr()))
    buf225 = empty((48, 197, 197), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_14], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf223, (48, 197, 64), (12608, 64, 1), 0), reinterpret_tensor(buf224, (48, 64, 197), (12608, 197, 1), 0), out=buf225)
    buf226 = buf173; del buf173  # reuse
    buf227 = reinterpret_tensor(buf225, (8, 6, 197, 197), (232854, 38809, 197, 1), 0); del buf225  # reuse
    buf228 = buf171; del buf171  # reuse
    buf229 = buf227; del buf227  # reuse
    buf230 = empty((8, 6, 197, 64), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_mul_47(c_void_p(buf229.data_ptr()), c_void_p(buf222.data_ptr()), c_void_p(buf226.data_ptr()), c_void_p(buf228.data_ptr()), c_void_p(buf230.data_ptr()))
    buf231 = reinterpret_tensor(buf222, (48, 197, 64), (12608, 64, 1), 0); del buf222  # reuse
    # Source Nodes: [matmul_15], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf229, (48, 197, 197), (38809, 197, 1), 0), reinterpret_tensor(buf230, (48, 197, 64), (12608, 64, 1), 0), out=buf231)
    buf232 = empty((1576, 384), device='cpu', dtype=torch.float32)
    cpp_fused_view_48(c_void_p(buf231.data_ptr()), c_void_p(buf232.data_ptr()))
    buf233 = reinterpret_tensor(buf231, (1576, 384), (384, 1), 0); del buf231  # reuse
    # Source Nodes: [x_69], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_117, buf232, reinterpret_tensor(primals_116, (384, 384), (1, 384), 0), alpha=1, beta=1, out=buf233)
    del primals_117
    buf234 = empty_strided((8, 197, 1), (197, 1, 1576), device='cpu', dtype=torch.float32)
    buf235 = empty_strided((8, 197, 1), (197, 1, 1576), device='cpu', dtype=torch.float32)
    buf237 = empty((8, 197, 384), device='cpu', dtype=torch.float32)
    buf238 = empty((1576, 384), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_49(c_void_p(buf215.data_ptr()), c_void_p(buf233.data_ptr()), c_void_p(primals_118.data_ptr()), c_void_p(primals_119.data_ptr()), c_void_p(buf234.data_ptr()), c_void_p(buf235.data_ptr()), c_void_p(buf237.data_ptr()), c_void_p(buf238.data_ptr()))
    del primals_119
    buf239 = empty((1576, 1536), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_71], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_121, buf238, reinterpret_tensor(primals_120, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf239)
    del primals_121
    buf240 = empty((1576, 1536), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_50(c_void_p(buf239.data_ptr()), c_void_p(buf240.data_ptr()))
    buf241 = empty((1576, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_75], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_123, buf240, reinterpret_tensor(primals_122, (1536, 384), (1, 1536), 0), alpha=1, beta=1, out=buf241)
    del primals_123
    buf243 = buf188; del buf188  # reuse
    # Source Nodes: [l__mod___blocks_4_attn_in_qk], Original ATen: [aten.mm]
    extern_kernels.mm(buf242, reinterpret_tensor(primals_126, (24, 48), (1, 24), 0), out=buf243)
    buf244 = reinterpret_tensor(buf214, (25088, 24), (24, 1), 0); del buf214  # reuse
    # Source Nodes: [l__mod___blocks_4_attn_in_v], Original ATen: [aten.mm]
    extern_kernels.mm(buf242, reinterpret_tensor(primals_127, (24, 24), (1, 24), 0), out=buf244)
    buf245 = empty((1568, 4, 16, 6), device='cpu', dtype=torch.float32)
    buf246 = empty((1568, 4, 6, 16), device='cpu', dtype=torch.float32)
    cpp_fused_clone_51(c_void_p(buf243.data_ptr()), c_void_p(buf245.data_ptr()), c_void_p(buf246.data_ptr()))
    buf247 = empty((6272, 16, 16), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_16], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf245, (6272, 16, 6), (96, 6, 1), 0), reinterpret_tensor(buf246, (6272, 6, 16), (96, 16, 1), 0), out=buf247)
    buf248 = buf195; del buf195  # reuse
    buf249 = reinterpret_tensor(buf247, (1568, 4, 16, 16), (1024, 256, 16, 1), 0); del buf247  # reuse
    buf250 = buf193; del buf193  # reuse
    buf251 = buf249; del buf249  # reuse
    buf252 = empty((1568, 4, 16, 6), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_mul_52(c_void_p(buf251.data_ptr()), c_void_p(buf244.data_ptr()), c_void_p(buf248.data_ptr()), c_void_p(buf250.data_ptr()), c_void_p(buf252.data_ptr()))
    buf253 = reinterpret_tensor(buf244, (6272, 16, 6), (96, 6, 1), 0); del buf244  # reuse
    # Source Nodes: [matmul_17], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf251, (6272, 16, 16), (256, 16, 1), 0), reinterpret_tensor(buf252, (6272, 16, 6), (96, 6, 1), 0), out=buf253)
    buf254 = empty((25088, 24), device='cpu', dtype=torch.float32)
    cpp_fused_view_53(c_void_p(buf253.data_ptr()), c_void_p(buf254.data_ptr()))
    buf255 = reinterpret_tensor(buf253, (25088, 24), (24, 1), 0); del buf253  # reuse
    # Source Nodes: [x_78], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_129, buf254, reinterpret_tensor(primals_128, (24, 24), (1, 24), 0), alpha=1, beta=1, out=buf255)
    del primals_129
    buf256 = reinterpret_tensor(buf255, (1568, 16, 24), (384, 24, 1), 0); del buf255  # reuse
    buf257 = buf209; del buf209  # reuse
    buf258 = empty_strided((1568, 16, 1), (16, 1, 25088), device='cpu', dtype=torch.float32)
    buf260 = empty((1568, 16, 24), device='cpu', dtype=torch.float32)
    buf261 = empty((25088, 24), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_54(c_void_p(buf256.data_ptr()), c_void_p(buf145.data_ptr()), c_void_p(buf153.data_ptr()), c_void_p(buf200.data_ptr()), c_void_p(buf208.data_ptr()), c_void_p(primals_130.data_ptr()), c_void_p(primals_131.data_ptr()), c_void_p(buf257.data_ptr()), c_void_p(buf258.data_ptr()), c_void_p(buf260.data_ptr()), c_void_p(buf261.data_ptr()))
    del primals_131
    buf262 = empty((25088, 96), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_80], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_133, buf261, reinterpret_tensor(primals_132, (24, 96), (1, 24), 0), alpha=1, beta=1, out=buf262)
    del primals_133
    buf263 = empty((25088, 96), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_55(c_void_p(buf262.data_ptr()), c_void_p(buf263.data_ptr()))
    buf264 = buf208; del buf208  # reuse
    # Source Nodes: [x_84], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_135, buf263, reinterpret_tensor(primals_134, (96, 24), (1, 96), 0), alpha=1, beta=1, out=buf264)
    del primals_135
    buf265 = buf257; del buf257  # reuse
    buf266 = empty_strided((1568, 16, 1), (16, 1, 25088), device='cpu', dtype=torch.float32)
    buf268 = reinterpret_tensor(buf200, (1568, 16, 24), (384, 24, 1), 0); del buf200  # reuse
    buf298 = buf153; del buf153  # reuse
    buf269 = reinterpret_tensor(buf145, (1568, 384), (384, 1), 0); del buf145  # reuse
    cpp_fused_add_native_layer_norm_view_56(c_void_p(buf256.data_ptr()), c_void_p(buf264.data_ptr()), c_void_p(primals_152.data_ptr()), c_void_p(primals_153.data_ptr()), c_void_p(primals_136.data_ptr()), c_void_p(primals_137.data_ptr()), c_void_p(buf265.data_ptr()), c_void_p(buf266.data_ptr()), c_void_p(buf268.data_ptr()), c_void_p(buf298.data_ptr()), c_void_p(buf269.data_ptr()))
    del primals_137
    del primals_153
    buf270 = empty((1568, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___blocks_4_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_139, buf269, reinterpret_tensor(primals_138, (384, 384), (1, 384), 0), alpha=1, beta=1, out=buf270)
    del primals_139
    buf271 = reinterpret_tensor(buf241, (8, 197, 384), (75648, 384, 1), 0); del buf241  # reuse
    buf272 = reinterpret_tensor(buf234, (8, 197, 1), (197, 1, 1), 0); del buf234  # reuse
    buf273 = empty_strided((8, 197, 1), (197, 1, 1576), device='cpu', dtype=torch.float32)
    buf275 = reinterpret_tensor(buf273, (8, 197, 1), (197, 1, 1), 0); del buf273  # reuse
    buf276 = empty((1576, 384), device='cpu', dtype=torch.float32)
    cpp_fused_cat_native_layer_norm_view_57(c_void_p(buf271.data_ptr()), c_void_p(buf275.data_ptr()), c_void_p(buf215.data_ptr()), c_void_p(buf233.data_ptr()), c_void_p(buf270.data_ptr()), c_void_p(primals_140.data_ptr()), c_void_p(primals_141.data_ptr()), c_void_p(buf272.data_ptr()), c_void_p(buf276.data_ptr()))
    del primals_141
    buf277 = buf221; del buf221  # reuse
    # Source Nodes: [l__mod___blocks_4_attn_out_qk], Original ATen: [aten.mm]
    extern_kernels.mm(buf276, reinterpret_tensor(primals_142, (384, 768), (1, 384), 0), out=buf277)
    buf278 = buf233; del buf233  # reuse
    # Source Nodes: [l__mod___blocks_4_attn_out_v], Original ATen: [aten.mm]
    extern_kernels.mm(buf276, reinterpret_tensor(primals_143, (384, 384), (1, 384), 0), out=buf278)
    buf279 = empty((8, 6, 197, 64), device='cpu', dtype=torch.float32)
    buf280 = empty((8, 6, 64, 197), device='cpu', dtype=torch.float32)
    cpp_fused_clone_58(c_void_p(buf277.data_ptr()), c_void_p(buf279.data_ptr()), c_void_p(buf280.data_ptr()))
    buf281 = empty((48, 197, 197), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_18], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf279, (48, 197, 64), (12608, 64, 1), 0), reinterpret_tensor(buf280, (48, 64, 197), (12608, 197, 1), 0), out=buf281)
    buf282 = buf228; del buf228  # reuse
    buf283 = reinterpret_tensor(buf281, (8, 6, 197, 197), (232854, 38809, 197, 1), 0); del buf281  # reuse
    buf284 = buf226; del buf226  # reuse
    buf285 = buf283; del buf283  # reuse
    buf286 = empty((8, 6, 197, 64), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_mul_59(c_void_p(buf285.data_ptr()), c_void_p(buf278.data_ptr()), c_void_p(buf282.data_ptr()), c_void_p(buf284.data_ptr()), c_void_p(buf286.data_ptr()))
    buf287 = reinterpret_tensor(buf278, (48, 197, 64), (12608, 64, 1), 0); del buf278  # reuse
    # Source Nodes: [matmul_19], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf285, (48, 197, 197), (38809, 197, 1), 0), reinterpret_tensor(buf286, (48, 197, 64), (12608, 64, 1), 0), out=buf287)
    buf288 = empty((1576, 384), device='cpu', dtype=torch.float32)
    cpp_fused_view_60(c_void_p(buf287.data_ptr()), c_void_p(buf288.data_ptr()))
    buf289 = reinterpret_tensor(buf287, (1576, 384), (384, 1), 0); del buf287  # reuse
    # Source Nodes: [x_87], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_145, buf288, reinterpret_tensor(primals_144, (384, 384), (1, 384), 0), alpha=1, beta=1, out=buf289)
    del primals_145
    buf290 = empty_strided((8, 197, 1), (197, 1, 1576), device='cpu', dtype=torch.float32)
    buf291 = empty_strided((8, 197, 1), (197, 1, 1576), device='cpu', dtype=torch.float32)
    buf293 = empty((8, 197, 384), device='cpu', dtype=torch.float32)
    buf294 = empty((1576, 384), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_61(c_void_p(buf271.data_ptr()), c_void_p(buf289.data_ptr()), c_void_p(primals_146.data_ptr()), c_void_p(primals_147.data_ptr()), c_void_p(buf290.data_ptr()), c_void_p(buf291.data_ptr()), c_void_p(buf293.data_ptr()), c_void_p(buf294.data_ptr()))
    del primals_147
    buf295 = empty((1576, 1536), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_89], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_149, buf294, reinterpret_tensor(primals_148, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf295)
    del primals_149
    buf296 = empty((1576, 1536), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_62(c_void_p(buf295.data_ptr()), c_void_p(buf296.data_ptr()))
    buf297 = empty((1576, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_93], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_151, buf296, reinterpret_tensor(primals_150, (1536, 384), (1, 1536), 0), alpha=1, beta=1, out=buf297)
    del primals_151
    buf299 = buf243; del buf243  # reuse
    # Source Nodes: [l__mod___blocks_5_attn_in_qk], Original ATen: [aten.mm]
    extern_kernels.mm(buf298, reinterpret_tensor(primals_154, (24, 48), (1, 24), 0), out=buf299)
    buf300 = reinterpret_tensor(buf270, (25088, 24), (24, 1), 0); del buf270  # reuse
    # Source Nodes: [l__mod___blocks_5_attn_in_v], Original ATen: [aten.mm]
    extern_kernels.mm(buf298, reinterpret_tensor(primals_155, (24, 24), (1, 24), 0), out=buf300)
    buf301 = empty((1568, 4, 16, 6), device='cpu', dtype=torch.float32)
    buf302 = empty((1568, 4, 6, 16), device='cpu', dtype=torch.float32)
    cpp_fused_clone_63(c_void_p(buf299.data_ptr()), c_void_p(buf301.data_ptr()), c_void_p(buf302.data_ptr()))
    buf303 = empty((6272, 16, 16), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_20], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf301, (6272, 16, 6), (96, 6, 1), 0), reinterpret_tensor(buf302, (6272, 6, 16), (96, 16, 1), 0), out=buf303)
    buf304 = buf250; del buf250  # reuse
    buf305 = reinterpret_tensor(buf303, (1568, 4, 16, 16), (1024, 256, 16, 1), 0); del buf303  # reuse
    buf306 = buf248; del buf248  # reuse
    buf307 = buf305; del buf305  # reuse
    buf308 = empty((1568, 4, 16, 6), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_mul_64(c_void_p(buf307.data_ptr()), c_void_p(buf300.data_ptr()), c_void_p(buf304.data_ptr()), c_void_p(buf306.data_ptr()), c_void_p(buf308.data_ptr()))
    buf309 = reinterpret_tensor(buf300, (6272, 16, 6), (96, 6, 1), 0); del buf300  # reuse
    # Source Nodes: [matmul_21], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf307, (6272, 16, 16), (256, 16, 1), 0), reinterpret_tensor(buf308, (6272, 16, 6), (96, 6, 1), 0), out=buf309)
    buf310 = empty((25088, 24), device='cpu', dtype=torch.float32)
    cpp_fused_view_65(c_void_p(buf309.data_ptr()), c_void_p(buf310.data_ptr()))
    buf311 = reinterpret_tensor(buf309, (25088, 24), (24, 1), 0); del buf309  # reuse
    # Source Nodes: [x_96], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_157, buf310, reinterpret_tensor(primals_156, (24, 24), (1, 24), 0), alpha=1, beta=1, out=buf311)
    del primals_157
    buf312 = buf265; del buf265  # reuse
    buf313 = empty_strided((1568, 16, 1), (16, 1, 25088), device='cpu', dtype=torch.float32)
    buf315 = empty((1568, 16, 24), device='cpu', dtype=torch.float32)
    buf316 = empty((25088, 24), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_66(c_void_p(buf256.data_ptr()), c_void_p(buf264.data_ptr()), c_void_p(buf311.data_ptr()), c_void_p(primals_158.data_ptr()), c_void_p(primals_159.data_ptr()), c_void_p(buf312.data_ptr()), c_void_p(buf313.data_ptr()), c_void_p(buf315.data_ptr()), c_void_p(buf316.data_ptr()))
    del primals_159
    buf317 = empty((25088, 96), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_98], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_161, buf316, reinterpret_tensor(primals_160, (24, 96), (1, 24), 0), alpha=1, beta=1, out=buf317)
    del primals_161
    buf318 = empty((25088, 96), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_67(c_void_p(buf317.data_ptr()), c_void_p(buf318.data_ptr()))
    buf319 = empty((25088, 24), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_102], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_163, buf318, reinterpret_tensor(primals_162, (96, 24), (1, 96), 0), alpha=1, beta=1, out=buf319)
    del primals_163
    buf320 = buf312; del buf312  # reuse
    buf321 = empty_strided((1568, 16, 1), (16, 1, 25088), device='cpu', dtype=torch.float32)
    buf323 = empty((1568, 16, 24), device='cpu', dtype=torch.float32)
    buf353 = empty((25088, 24), device='cpu', dtype=torch.float32)
    buf324 = empty((1568, 384), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_68(c_void_p(buf256.data_ptr()), c_void_p(buf264.data_ptr()), c_void_p(buf311.data_ptr()), c_void_p(buf319.data_ptr()), c_void_p(primals_180.data_ptr()), c_void_p(primals_181.data_ptr()), c_void_p(primals_164.data_ptr()), c_void_p(primals_165.data_ptr()), c_void_p(buf320.data_ptr()), c_void_p(buf321.data_ptr()), c_void_p(buf323.data_ptr()), c_void_p(buf353.data_ptr()), c_void_p(buf324.data_ptr()))
    del primals_165
    del primals_181
    buf325 = empty((1568, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___blocks_5_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_167, buf324, reinterpret_tensor(primals_166, (384, 384), (1, 384), 0), alpha=1, beta=1, out=buf325)
    del primals_167
    buf326 = reinterpret_tensor(buf297, (8, 197, 384), (75648, 384, 1), 0); del buf297  # reuse
    buf327 = reinterpret_tensor(buf290, (8, 197, 1), (197, 1, 1), 0); del buf290  # reuse
    buf328 = empty_strided((8, 197, 1), (197, 1, 1576), device='cpu', dtype=torch.float32)
    buf330 = reinterpret_tensor(buf328, (8, 197, 1), (197, 1, 1), 0); del buf328  # reuse
    buf331 = empty((1576, 384), device='cpu', dtype=torch.float32)
    cpp_fused_cat_native_layer_norm_view_69(c_void_p(buf326.data_ptr()), c_void_p(buf330.data_ptr()), c_void_p(buf271.data_ptr()), c_void_p(buf289.data_ptr()), c_void_p(buf325.data_ptr()), c_void_p(primals_168.data_ptr()), c_void_p(primals_169.data_ptr()), c_void_p(buf327.data_ptr()), c_void_p(buf331.data_ptr()))
    del primals_169
    buf332 = buf277; del buf277  # reuse
    # Source Nodes: [l__mod___blocks_5_attn_out_qk], Original ATen: [aten.mm]
    extern_kernels.mm(buf331, reinterpret_tensor(primals_170, (384, 768), (1, 384), 0), out=buf332)
    buf333 = buf289; del buf289  # reuse
    # Source Nodes: [l__mod___blocks_5_attn_out_v], Original ATen: [aten.mm]
    extern_kernels.mm(buf331, reinterpret_tensor(primals_171, (384, 384), (1, 384), 0), out=buf333)
    buf334 = empty((8, 6, 197, 64), device='cpu', dtype=torch.float32)
    buf335 = empty((8, 6, 64, 197), device='cpu', dtype=torch.float32)
    cpp_fused_clone_70(c_void_p(buf332.data_ptr()), c_void_p(buf334.data_ptr()), c_void_p(buf335.data_ptr()))
    buf336 = empty((48, 197, 197), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_22], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf334, (48, 197, 64), (12608, 64, 1), 0), reinterpret_tensor(buf335, (48, 64, 197), (12608, 197, 1), 0), out=buf336)
    buf337 = buf284; del buf284  # reuse
    buf338 = reinterpret_tensor(buf336, (8, 6, 197, 197), (232854, 38809, 197, 1), 0); del buf336  # reuse
    buf339 = buf282; del buf282  # reuse
    buf340 = buf338; del buf338  # reuse
    buf341 = empty((8, 6, 197, 64), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_mul_71(c_void_p(buf340.data_ptr()), c_void_p(buf333.data_ptr()), c_void_p(buf337.data_ptr()), c_void_p(buf339.data_ptr()), c_void_p(buf341.data_ptr()))
    buf342 = reinterpret_tensor(buf333, (48, 197, 64), (12608, 64, 1), 0); del buf333  # reuse
    # Source Nodes: [matmul_23], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf340, (48, 197, 197), (38809, 197, 1), 0), reinterpret_tensor(buf341, (48, 197, 64), (12608, 64, 1), 0), out=buf342)
    buf343 = empty((1576, 384), device='cpu', dtype=torch.float32)
    cpp_fused_view_72(c_void_p(buf342.data_ptr()), c_void_p(buf343.data_ptr()))
    buf344 = reinterpret_tensor(buf342, (1576, 384), (384, 1), 0); del buf342  # reuse
    # Source Nodes: [x_105], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_173, buf343, reinterpret_tensor(primals_172, (384, 384), (1, 384), 0), alpha=1, beta=1, out=buf344)
    del primals_173
    buf345 = empty_strided((8, 197, 1), (197, 1, 1576), device='cpu', dtype=torch.float32)
    buf346 = empty_strided((8, 197, 1), (197, 1, 1576), device='cpu', dtype=torch.float32)
    buf348 = empty((8, 197, 384), device='cpu', dtype=torch.float32)
    buf349 = empty((1576, 384), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_73(c_void_p(buf326.data_ptr()), c_void_p(buf344.data_ptr()), c_void_p(primals_174.data_ptr()), c_void_p(primals_175.data_ptr()), c_void_p(buf345.data_ptr()), c_void_p(buf346.data_ptr()), c_void_p(buf348.data_ptr()), c_void_p(buf349.data_ptr()))
    del primals_175
    buf350 = empty((1576, 1536), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_107], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_177, buf349, reinterpret_tensor(primals_176, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf350)
    del primals_177
    buf351 = empty((1576, 1536), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_74(c_void_p(buf350.data_ptr()), c_void_p(buf351.data_ptr()))
    buf352 = empty((1576, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_111], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_179, buf351, reinterpret_tensor(primals_178, (1536, 384), (1, 1536), 0), alpha=1, beta=1, out=buf352)
    del primals_179
    buf354 = buf299; del buf299  # reuse
    # Source Nodes: [l__mod___blocks_6_attn_in_qk], Original ATen: [aten.mm]
    extern_kernels.mm(buf353, reinterpret_tensor(primals_182, (24, 48), (1, 24), 0), out=buf354)
    buf355 = reinterpret_tensor(buf325, (25088, 24), (24, 1), 0); del buf325  # reuse
    # Source Nodes: [l__mod___blocks_6_attn_in_v], Original ATen: [aten.mm]
    extern_kernels.mm(buf353, reinterpret_tensor(primals_183, (24, 24), (1, 24), 0), out=buf355)
    buf356 = empty((1568, 4, 16, 6), device='cpu', dtype=torch.float32)
    buf357 = empty((1568, 4, 6, 16), device='cpu', dtype=torch.float32)
    cpp_fused_clone_75(c_void_p(buf354.data_ptr()), c_void_p(buf356.data_ptr()), c_void_p(buf357.data_ptr()))
    buf358 = empty((6272, 16, 16), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_24], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf356, (6272, 16, 6), (96, 6, 1), 0), reinterpret_tensor(buf357, (6272, 6, 16), (96, 16, 1), 0), out=buf358)
    buf359 = buf306; del buf306  # reuse
    buf360 = reinterpret_tensor(buf358, (1568, 4, 16, 16), (1024, 256, 16, 1), 0); del buf358  # reuse
    buf361 = buf304; del buf304  # reuse
    buf362 = buf360; del buf360  # reuse
    buf363 = empty((1568, 4, 16, 6), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_mul_76(c_void_p(buf362.data_ptr()), c_void_p(buf355.data_ptr()), c_void_p(buf359.data_ptr()), c_void_p(buf361.data_ptr()), c_void_p(buf363.data_ptr()))
    buf364 = reinterpret_tensor(buf355, (6272, 16, 6), (96, 6, 1), 0); del buf355  # reuse
    # Source Nodes: [matmul_25], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf362, (6272, 16, 16), (256, 16, 1), 0), reinterpret_tensor(buf363, (6272, 16, 6), (96, 6, 1), 0), out=buf364)
    buf365 = empty((25088, 24), device='cpu', dtype=torch.float32)
    cpp_fused_view_77(c_void_p(buf364.data_ptr()), c_void_p(buf365.data_ptr()))
    buf366 = reinterpret_tensor(buf364, (25088, 24), (24, 1), 0); del buf364  # reuse
    # Source Nodes: [x_114], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_185, buf365, reinterpret_tensor(primals_184, (24, 24), (1, 24), 0), alpha=1, beta=1, out=buf366)
    del primals_185
    buf367 = reinterpret_tensor(buf366, (1568, 16, 24), (384, 24, 1), 0); del buf366  # reuse
    buf368 = buf320; del buf320  # reuse
    buf369 = empty_strided((1568, 16, 1), (16, 1, 25088), device='cpu', dtype=torch.float32)
    buf371 = empty((1568, 16, 24), device='cpu', dtype=torch.float32)
    buf372 = empty((25088, 24), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_78(c_void_p(buf367.data_ptr()), c_void_p(buf256.data_ptr()), c_void_p(buf264.data_ptr()), c_void_p(buf311.data_ptr()), c_void_p(buf319.data_ptr()), c_void_p(primals_186.data_ptr()), c_void_p(primals_187.data_ptr()), c_void_p(buf368.data_ptr()), c_void_p(buf369.data_ptr()), c_void_p(buf371.data_ptr()), c_void_p(buf372.data_ptr()))
    del primals_187
    buf373 = empty((25088, 96), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_116], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_189, buf372, reinterpret_tensor(primals_188, (24, 96), (1, 24), 0), alpha=1, beta=1, out=buf373)
    del primals_189
    buf374 = empty((25088, 96), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_79(c_void_p(buf373.data_ptr()), c_void_p(buf374.data_ptr()))
    buf375 = buf319; del buf319  # reuse
    # Source Nodes: [x_120], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_191, buf374, reinterpret_tensor(primals_190, (96, 24), (1, 96), 0), alpha=1, beta=1, out=buf375)
    del primals_191
    buf376 = buf368; del buf368  # reuse
    buf377 = empty_strided((1568, 16, 1), (16, 1, 25088), device='cpu', dtype=torch.float32)
    buf379 = reinterpret_tensor(buf311, (1568, 16, 24), (384, 24, 1), 0); del buf311  # reuse
    buf409 = buf264; del buf264  # reuse
    buf380 = reinterpret_tensor(buf256, (1568, 384), (384, 1), 0); del buf256  # reuse
    cpp_fused_add_native_layer_norm_view_80(c_void_p(buf367.data_ptr()), c_void_p(buf375.data_ptr()), c_void_p(primals_208.data_ptr()), c_void_p(primals_209.data_ptr()), c_void_p(primals_192.data_ptr()), c_void_p(primals_193.data_ptr()), c_void_p(buf376.data_ptr()), c_void_p(buf377.data_ptr()), c_void_p(buf379.data_ptr()), c_void_p(buf409.data_ptr()), c_void_p(buf380.data_ptr()))
    del primals_193
    del primals_209
    buf381 = empty((1568, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___blocks_6_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_195, buf380, reinterpret_tensor(primals_194, (384, 384), (1, 384), 0), alpha=1, beta=1, out=buf381)
    del primals_195
    buf382 = reinterpret_tensor(buf352, (8, 197, 384), (75648, 384, 1), 0); del buf352  # reuse
    buf383 = reinterpret_tensor(buf345, (8, 197, 1), (197, 1, 1), 0); del buf345  # reuse
    buf384 = empty_strided((8, 197, 1), (197, 1, 1576), device='cpu', dtype=torch.float32)
    buf386 = reinterpret_tensor(buf384, (8, 197, 1), (197, 1, 1), 0); del buf384  # reuse
    buf387 = empty((1576, 384), device='cpu', dtype=torch.float32)
    cpp_fused_cat_native_layer_norm_view_81(c_void_p(buf382.data_ptr()), c_void_p(buf386.data_ptr()), c_void_p(buf326.data_ptr()), c_void_p(buf344.data_ptr()), c_void_p(buf381.data_ptr()), c_void_p(primals_196.data_ptr()), c_void_p(primals_197.data_ptr()), c_void_p(buf383.data_ptr()), c_void_p(buf387.data_ptr()))
    del primals_197
    buf388 = buf332; del buf332  # reuse
    # Source Nodes: [l__mod___blocks_6_attn_out_qk], Original ATen: [aten.mm]
    extern_kernels.mm(buf387, reinterpret_tensor(primals_198, (384, 768), (1, 384), 0), out=buf388)
    buf389 = buf344; del buf344  # reuse
    # Source Nodes: [l__mod___blocks_6_attn_out_v], Original ATen: [aten.mm]
    extern_kernels.mm(buf387, reinterpret_tensor(primals_199, (384, 384), (1, 384), 0), out=buf389)
    buf390 = empty((8, 6, 197, 64), device='cpu', dtype=torch.float32)
    buf391 = empty((8, 6, 64, 197), device='cpu', dtype=torch.float32)
    cpp_fused_clone_82(c_void_p(buf388.data_ptr()), c_void_p(buf390.data_ptr()), c_void_p(buf391.data_ptr()))
    buf392 = empty((48, 197, 197), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_26], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf390, (48, 197, 64), (12608, 64, 1), 0), reinterpret_tensor(buf391, (48, 64, 197), (12608, 197, 1), 0), out=buf392)
    buf393 = buf339; del buf339  # reuse
    buf394 = reinterpret_tensor(buf392, (8, 6, 197, 197), (232854, 38809, 197, 1), 0); del buf392  # reuse
    buf395 = buf337; del buf337  # reuse
    buf396 = buf394; del buf394  # reuse
    buf397 = empty((8, 6, 197, 64), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_mul_83(c_void_p(buf396.data_ptr()), c_void_p(buf389.data_ptr()), c_void_p(buf393.data_ptr()), c_void_p(buf395.data_ptr()), c_void_p(buf397.data_ptr()))
    buf398 = reinterpret_tensor(buf389, (48, 197, 64), (12608, 64, 1), 0); del buf389  # reuse
    # Source Nodes: [matmul_27], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf396, (48, 197, 197), (38809, 197, 1), 0), reinterpret_tensor(buf397, (48, 197, 64), (12608, 64, 1), 0), out=buf398)
    buf399 = empty((1576, 384), device='cpu', dtype=torch.float32)
    cpp_fused_view_84(c_void_p(buf398.data_ptr()), c_void_p(buf399.data_ptr()))
    buf400 = reinterpret_tensor(buf398, (1576, 384), (384, 1), 0); del buf398  # reuse
    # Source Nodes: [x_123], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_201, buf399, reinterpret_tensor(primals_200, (384, 384), (1, 384), 0), alpha=1, beta=1, out=buf400)
    del primals_201
    buf401 = empty_strided((8, 197, 1), (197, 1, 1576), device='cpu', dtype=torch.float32)
    buf402 = empty_strided((8, 197, 1), (197, 1, 1576), device='cpu', dtype=torch.float32)
    buf404 = empty((8, 197, 384), device='cpu', dtype=torch.float32)
    buf405 = empty((1576, 384), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_85(c_void_p(buf382.data_ptr()), c_void_p(buf400.data_ptr()), c_void_p(primals_202.data_ptr()), c_void_p(primals_203.data_ptr()), c_void_p(buf401.data_ptr()), c_void_p(buf402.data_ptr()), c_void_p(buf404.data_ptr()), c_void_p(buf405.data_ptr()))
    del primals_203
    buf406 = empty((1576, 1536), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_125], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_205, buf405, reinterpret_tensor(primals_204, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf406)
    del primals_205
    buf407 = empty((1576, 1536), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_86(c_void_p(buf406.data_ptr()), c_void_p(buf407.data_ptr()))
    buf408 = empty((1576, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_129], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_207, buf407, reinterpret_tensor(primals_206, (1536, 384), (1, 1536), 0), alpha=1, beta=1, out=buf408)
    del primals_207
    buf410 = buf354; del buf354  # reuse
    # Source Nodes: [l__mod___blocks_7_attn_in_qk], Original ATen: [aten.mm]
    extern_kernels.mm(buf409, reinterpret_tensor(primals_210, (24, 48), (1, 24), 0), out=buf410)
    buf411 = reinterpret_tensor(buf381, (25088, 24), (24, 1), 0); del buf381  # reuse
    # Source Nodes: [l__mod___blocks_7_attn_in_v], Original ATen: [aten.mm]
    extern_kernels.mm(buf409, reinterpret_tensor(primals_211, (24, 24), (1, 24), 0), out=buf411)
    buf412 = empty((1568, 4, 16, 6), device='cpu', dtype=torch.float32)
    buf413 = empty((1568, 4, 6, 16), device='cpu', dtype=torch.float32)
    cpp_fused_clone_87(c_void_p(buf410.data_ptr()), c_void_p(buf412.data_ptr()), c_void_p(buf413.data_ptr()))
    buf414 = empty((6272, 16, 16), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_28], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf412, (6272, 16, 6), (96, 6, 1), 0), reinterpret_tensor(buf413, (6272, 6, 16), (96, 16, 1), 0), out=buf414)
    buf415 = buf361; del buf361  # reuse
    buf416 = reinterpret_tensor(buf414, (1568, 4, 16, 16), (1024, 256, 16, 1), 0); del buf414  # reuse
    buf417 = buf359; del buf359  # reuse
    buf418 = buf416; del buf416  # reuse
    buf419 = empty((1568, 4, 16, 6), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_mul_88(c_void_p(buf418.data_ptr()), c_void_p(buf411.data_ptr()), c_void_p(buf415.data_ptr()), c_void_p(buf417.data_ptr()), c_void_p(buf419.data_ptr()))
    buf420 = reinterpret_tensor(buf411, (6272, 16, 6), (96, 6, 1), 0); del buf411  # reuse
    # Source Nodes: [matmul_29], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf418, (6272, 16, 16), (256, 16, 1), 0), reinterpret_tensor(buf419, (6272, 16, 6), (96, 6, 1), 0), out=buf420)
    buf421 = empty((25088, 24), device='cpu', dtype=torch.float32)
    cpp_fused_view_89(c_void_p(buf420.data_ptr()), c_void_p(buf421.data_ptr()))
    buf422 = reinterpret_tensor(buf420, (25088, 24), (24, 1), 0); del buf420  # reuse
    # Source Nodes: [x_132], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_213, buf421, reinterpret_tensor(primals_212, (24, 24), (1, 24), 0), alpha=1, beta=1, out=buf422)
    del primals_213
    buf423 = buf376; del buf376  # reuse
    buf424 = empty_strided((1568, 16, 1), (16, 1, 25088), device='cpu', dtype=torch.float32)
    buf426 = empty((1568, 16, 24), device='cpu', dtype=torch.float32)
    buf427 = empty((25088, 24), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_90(c_void_p(buf367.data_ptr()), c_void_p(buf375.data_ptr()), c_void_p(buf422.data_ptr()), c_void_p(primals_214.data_ptr()), c_void_p(primals_215.data_ptr()), c_void_p(buf423.data_ptr()), c_void_p(buf424.data_ptr()), c_void_p(buf426.data_ptr()), c_void_p(buf427.data_ptr()))
    del primals_215
    buf428 = empty((25088, 96), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_134], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_217, buf427, reinterpret_tensor(primals_216, (24, 96), (1, 24), 0), alpha=1, beta=1, out=buf428)
    del primals_217
    buf429 = empty((25088, 96), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_91(c_void_p(buf428.data_ptr()), c_void_p(buf429.data_ptr()))
    buf430 = empty((25088, 24), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_138], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_219, buf429, reinterpret_tensor(primals_218, (96, 24), (1, 96), 0), alpha=1, beta=1, out=buf430)
    del primals_219
    buf431 = buf423; del buf423  # reuse
    buf432 = empty_strided((1568, 16, 1), (16, 1, 25088), device='cpu', dtype=torch.float32)
    buf434 = empty((1568, 16, 24), device='cpu', dtype=torch.float32)
    buf464 = empty((25088, 24), device='cpu', dtype=torch.float32)
    buf435 = empty((1568, 384), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_92(c_void_p(buf367.data_ptr()), c_void_p(buf375.data_ptr()), c_void_p(buf422.data_ptr()), c_void_p(buf430.data_ptr()), c_void_p(primals_236.data_ptr()), c_void_p(primals_237.data_ptr()), c_void_p(primals_220.data_ptr()), c_void_p(primals_221.data_ptr()), c_void_p(buf431.data_ptr()), c_void_p(buf432.data_ptr()), c_void_p(buf434.data_ptr()), c_void_p(buf464.data_ptr()), c_void_p(buf435.data_ptr()))
    del primals_221
    del primals_237
    buf436 = empty((1568, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___blocks_7_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_223, buf435, reinterpret_tensor(primals_222, (384, 384), (1, 384), 0), alpha=1, beta=1, out=buf436)
    del primals_223
    buf437 = reinterpret_tensor(buf408, (8, 197, 384), (75648, 384, 1), 0); del buf408  # reuse
    buf438 = reinterpret_tensor(buf401, (8, 197, 1), (197, 1, 1), 0); del buf401  # reuse
    buf439 = empty_strided((8, 197, 1), (197, 1, 1576), device='cpu', dtype=torch.float32)
    buf441 = reinterpret_tensor(buf439, (8, 197, 1), (197, 1, 1), 0); del buf439  # reuse
    buf442 = empty((1576, 384), device='cpu', dtype=torch.float32)
    cpp_fused_cat_native_layer_norm_view_93(c_void_p(buf437.data_ptr()), c_void_p(buf441.data_ptr()), c_void_p(buf382.data_ptr()), c_void_p(buf400.data_ptr()), c_void_p(buf436.data_ptr()), c_void_p(primals_224.data_ptr()), c_void_p(primals_225.data_ptr()), c_void_p(buf438.data_ptr()), c_void_p(buf442.data_ptr()))
    del primals_225
    buf443 = buf388; del buf388  # reuse
    # Source Nodes: [l__mod___blocks_7_attn_out_qk], Original ATen: [aten.mm]
    extern_kernels.mm(buf442, reinterpret_tensor(primals_226, (384, 768), (1, 384), 0), out=buf443)
    buf444 = buf400; del buf400  # reuse
    # Source Nodes: [l__mod___blocks_7_attn_out_v], Original ATen: [aten.mm]
    extern_kernels.mm(buf442, reinterpret_tensor(primals_227, (384, 384), (1, 384), 0), out=buf444)
    buf445 = empty((8, 6, 197, 64), device='cpu', dtype=torch.float32)
    buf446 = empty((8, 6, 64, 197), device='cpu', dtype=torch.float32)
    cpp_fused_clone_94(c_void_p(buf443.data_ptr()), c_void_p(buf445.data_ptr()), c_void_p(buf446.data_ptr()))
    buf447 = empty((48, 197, 197), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_30], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf445, (48, 197, 64), (12608, 64, 1), 0), reinterpret_tensor(buf446, (48, 64, 197), (12608, 197, 1), 0), out=buf447)
    buf448 = buf395; del buf395  # reuse
    buf449 = reinterpret_tensor(buf447, (8, 6, 197, 197), (232854, 38809, 197, 1), 0); del buf447  # reuse
    buf450 = buf393; del buf393  # reuse
    buf451 = buf449; del buf449  # reuse
    buf452 = empty((8, 6, 197, 64), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_mul_95(c_void_p(buf451.data_ptr()), c_void_p(buf444.data_ptr()), c_void_p(buf448.data_ptr()), c_void_p(buf450.data_ptr()), c_void_p(buf452.data_ptr()))
    buf453 = reinterpret_tensor(buf444, (48, 197, 64), (12608, 64, 1), 0); del buf444  # reuse
    # Source Nodes: [matmul_31], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf451, (48, 197, 197), (38809, 197, 1), 0), reinterpret_tensor(buf452, (48, 197, 64), (12608, 64, 1), 0), out=buf453)
    buf454 = empty((1576, 384), device='cpu', dtype=torch.float32)
    cpp_fused_view_96(c_void_p(buf453.data_ptr()), c_void_p(buf454.data_ptr()))
    buf455 = reinterpret_tensor(buf453, (1576, 384), (384, 1), 0); del buf453  # reuse
    # Source Nodes: [x_141], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_229, buf454, reinterpret_tensor(primals_228, (384, 384), (1, 384), 0), alpha=1, beta=1, out=buf455)
    del primals_229
    buf456 = empty_strided((8, 197, 1), (197, 1, 1576), device='cpu', dtype=torch.float32)
    buf457 = empty_strided((8, 197, 1), (197, 1, 1576), device='cpu', dtype=torch.float32)
    buf459 = empty((8, 197, 384), device='cpu', dtype=torch.float32)
    buf460 = empty((1576, 384), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_97(c_void_p(buf437.data_ptr()), c_void_p(buf455.data_ptr()), c_void_p(primals_230.data_ptr()), c_void_p(primals_231.data_ptr()), c_void_p(buf456.data_ptr()), c_void_p(buf457.data_ptr()), c_void_p(buf459.data_ptr()), c_void_p(buf460.data_ptr()))
    del primals_231
    buf461 = empty((1576, 1536), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_143], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_233, buf460, reinterpret_tensor(primals_232, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf461)
    del primals_233
    buf462 = empty((1576, 1536), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_98(c_void_p(buf461.data_ptr()), c_void_p(buf462.data_ptr()))
    buf463 = empty((1576, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_147], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_235, buf462, reinterpret_tensor(primals_234, (1536, 384), (1, 1536), 0), alpha=1, beta=1, out=buf463)
    del primals_235
    buf465 = buf410; del buf410  # reuse
    # Source Nodes: [l__mod___blocks_8_attn_in_qk], Original ATen: [aten.mm]
    extern_kernels.mm(buf464, reinterpret_tensor(primals_238, (24, 48), (1, 24), 0), out=buf465)
    buf466 = reinterpret_tensor(buf436, (25088, 24), (24, 1), 0); del buf436  # reuse
    # Source Nodes: [l__mod___blocks_8_attn_in_v], Original ATen: [aten.mm]
    extern_kernels.mm(buf464, reinterpret_tensor(primals_239, (24, 24), (1, 24), 0), out=buf466)
    buf467 = empty((1568, 4, 16, 6), device='cpu', dtype=torch.float32)
    buf468 = empty((1568, 4, 6, 16), device='cpu', dtype=torch.float32)
    cpp_fused_clone_99(c_void_p(buf465.data_ptr()), c_void_p(buf467.data_ptr()), c_void_p(buf468.data_ptr()))
    buf469 = empty((6272, 16, 16), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_32], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf467, (6272, 16, 6), (96, 6, 1), 0), reinterpret_tensor(buf468, (6272, 6, 16), (96, 16, 1), 0), out=buf469)
    buf470 = buf417; del buf417  # reuse
    buf471 = reinterpret_tensor(buf469, (1568, 4, 16, 16), (1024, 256, 16, 1), 0); del buf469  # reuse
    buf472 = buf415; del buf415  # reuse
    buf473 = buf471; del buf471  # reuse
    buf474 = empty((1568, 4, 16, 6), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_mul_100(c_void_p(buf473.data_ptr()), c_void_p(buf466.data_ptr()), c_void_p(buf470.data_ptr()), c_void_p(buf472.data_ptr()), c_void_p(buf474.data_ptr()))
    buf475 = reinterpret_tensor(buf466, (6272, 16, 6), (96, 6, 1), 0); del buf466  # reuse
    # Source Nodes: [matmul_33], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf473, (6272, 16, 16), (256, 16, 1), 0), reinterpret_tensor(buf474, (6272, 16, 6), (96, 6, 1), 0), out=buf475)
    buf476 = empty((25088, 24), device='cpu', dtype=torch.float32)
    cpp_fused_view_101(c_void_p(buf475.data_ptr()), c_void_p(buf476.data_ptr()))
    buf477 = reinterpret_tensor(buf475, (25088, 24), (24, 1), 0); del buf475  # reuse
    # Source Nodes: [x_150], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_241, buf476, reinterpret_tensor(primals_240, (24, 24), (1, 24), 0), alpha=1, beta=1, out=buf477)
    del primals_241
    buf478 = reinterpret_tensor(buf477, (1568, 16, 24), (384, 24, 1), 0); del buf477  # reuse
    buf479 = buf431; del buf431  # reuse
    buf480 = empty_strided((1568, 16, 1), (16, 1, 25088), device='cpu', dtype=torch.float32)
    buf482 = empty((1568, 16, 24), device='cpu', dtype=torch.float32)
    buf483 = empty((25088, 24), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_102(c_void_p(buf478.data_ptr()), c_void_p(buf367.data_ptr()), c_void_p(buf375.data_ptr()), c_void_p(buf422.data_ptr()), c_void_p(buf430.data_ptr()), c_void_p(primals_242.data_ptr()), c_void_p(primals_243.data_ptr()), c_void_p(buf479.data_ptr()), c_void_p(buf480.data_ptr()), c_void_p(buf482.data_ptr()), c_void_p(buf483.data_ptr()))
    del primals_243
    buf484 = empty((25088, 96), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_152], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_245, buf483, reinterpret_tensor(primals_244, (24, 96), (1, 24), 0), alpha=1, beta=1, out=buf484)
    del primals_245
    buf485 = empty((25088, 96), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_103(c_void_p(buf484.data_ptr()), c_void_p(buf485.data_ptr()))
    buf486 = buf430; del buf430  # reuse
    # Source Nodes: [x_156], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_247, buf485, reinterpret_tensor(primals_246, (96, 24), (1, 96), 0), alpha=1, beta=1, out=buf486)
    del primals_247
    buf487 = buf479; del buf479  # reuse
    buf488 = empty_strided((1568, 16, 1), (16, 1, 25088), device='cpu', dtype=torch.float32)
    buf490 = reinterpret_tensor(buf422, (1568, 16, 24), (384, 24, 1), 0); del buf422  # reuse
    buf520 = buf375; del buf375  # reuse
    buf491 = reinterpret_tensor(buf367, (1568, 384), (384, 1), 0); del buf367  # reuse
    cpp_fused_add_native_layer_norm_view_104(c_void_p(buf478.data_ptr()), c_void_p(buf486.data_ptr()), c_void_p(primals_264.data_ptr()), c_void_p(primals_265.data_ptr()), c_void_p(primals_248.data_ptr()), c_void_p(primals_249.data_ptr()), c_void_p(buf487.data_ptr()), c_void_p(buf488.data_ptr()), c_void_p(buf490.data_ptr()), c_void_p(buf520.data_ptr()), c_void_p(buf491.data_ptr()))
    del primals_249
    del primals_265
    buf492 = empty((1568, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___blocks_8_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_251, buf491, reinterpret_tensor(primals_250, (384, 384), (1, 384), 0), alpha=1, beta=1, out=buf492)
    del primals_251
    buf493 = reinterpret_tensor(buf463, (8, 197, 384), (75648, 384, 1), 0); del buf463  # reuse
    buf494 = reinterpret_tensor(buf456, (8, 197, 1), (197, 1, 1), 0); del buf456  # reuse
    buf495 = empty_strided((8, 197, 1), (197, 1, 1576), device='cpu', dtype=torch.float32)
    buf497 = reinterpret_tensor(buf495, (8, 197, 1), (197, 1, 1), 0); del buf495  # reuse
    buf498 = empty((1576, 384), device='cpu', dtype=torch.float32)
    cpp_fused_cat_native_layer_norm_view_105(c_void_p(buf493.data_ptr()), c_void_p(buf497.data_ptr()), c_void_p(buf437.data_ptr()), c_void_p(buf455.data_ptr()), c_void_p(buf492.data_ptr()), c_void_p(primals_252.data_ptr()), c_void_p(primals_253.data_ptr()), c_void_p(buf494.data_ptr()), c_void_p(buf498.data_ptr()))
    del primals_253
    buf499 = buf443; del buf443  # reuse
    # Source Nodes: [l__mod___blocks_8_attn_out_qk], Original ATen: [aten.mm]
    extern_kernels.mm(buf498, reinterpret_tensor(primals_254, (384, 768), (1, 384), 0), out=buf499)
    buf500 = buf455; del buf455  # reuse
    # Source Nodes: [l__mod___blocks_8_attn_out_v], Original ATen: [aten.mm]
    extern_kernels.mm(buf498, reinterpret_tensor(primals_255, (384, 384), (1, 384), 0), out=buf500)
    buf501 = empty((8, 6, 197, 64), device='cpu', dtype=torch.float32)
    buf502 = empty((8, 6, 64, 197), device='cpu', dtype=torch.float32)
    cpp_fused_clone_106(c_void_p(buf499.data_ptr()), c_void_p(buf501.data_ptr()), c_void_p(buf502.data_ptr()))
    buf503 = empty((48, 197, 197), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_34], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf501, (48, 197, 64), (12608, 64, 1), 0), reinterpret_tensor(buf502, (48, 64, 197), (12608, 197, 1), 0), out=buf503)
    buf504 = buf450; del buf450  # reuse
    buf505 = reinterpret_tensor(buf503, (8, 6, 197, 197), (232854, 38809, 197, 1), 0); del buf503  # reuse
    buf506 = buf448; del buf448  # reuse
    buf507 = buf505; del buf505  # reuse
    buf508 = empty((8, 6, 197, 64), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_mul_107(c_void_p(buf507.data_ptr()), c_void_p(buf500.data_ptr()), c_void_p(buf504.data_ptr()), c_void_p(buf506.data_ptr()), c_void_p(buf508.data_ptr()))
    buf509 = reinterpret_tensor(buf500, (48, 197, 64), (12608, 64, 1), 0); del buf500  # reuse
    # Source Nodes: [matmul_35], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf507, (48, 197, 197), (38809, 197, 1), 0), reinterpret_tensor(buf508, (48, 197, 64), (12608, 64, 1), 0), out=buf509)
    buf510 = empty((1576, 384), device='cpu', dtype=torch.float32)
    cpp_fused_view_108(c_void_p(buf509.data_ptr()), c_void_p(buf510.data_ptr()))
    buf511 = reinterpret_tensor(buf509, (1576, 384), (384, 1), 0); del buf509  # reuse
    # Source Nodes: [x_159], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_257, buf510, reinterpret_tensor(primals_256, (384, 384), (1, 384), 0), alpha=1, beta=1, out=buf511)
    del primals_257
    buf512 = empty_strided((8, 197, 1), (197, 1, 1576), device='cpu', dtype=torch.float32)
    buf513 = empty_strided((8, 197, 1), (197, 1, 1576), device='cpu', dtype=torch.float32)
    buf515 = empty((8, 197, 384), device='cpu', dtype=torch.float32)
    buf516 = empty((1576, 384), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_109(c_void_p(buf493.data_ptr()), c_void_p(buf511.data_ptr()), c_void_p(primals_258.data_ptr()), c_void_p(primals_259.data_ptr()), c_void_p(buf512.data_ptr()), c_void_p(buf513.data_ptr()), c_void_p(buf515.data_ptr()), c_void_p(buf516.data_ptr()))
    del primals_259
    buf517 = empty((1576, 1536), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_161], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_261, buf516, reinterpret_tensor(primals_260, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf517)
    del primals_261
    buf518 = empty((1576, 1536), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_110(c_void_p(buf517.data_ptr()), c_void_p(buf518.data_ptr()))
    buf519 = empty((1576, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_165], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_263, buf518, reinterpret_tensor(primals_262, (1536, 384), (1, 1536), 0), alpha=1, beta=1, out=buf519)
    del primals_263
    buf521 = buf465; del buf465  # reuse
    # Source Nodes: [l__mod___blocks_9_attn_in_qk], Original ATen: [aten.mm]
    extern_kernels.mm(buf520, reinterpret_tensor(primals_266, (24, 48), (1, 24), 0), out=buf521)
    buf522 = reinterpret_tensor(buf492, (25088, 24), (24, 1), 0); del buf492  # reuse
    # Source Nodes: [l__mod___blocks_9_attn_in_v], Original ATen: [aten.mm]
    extern_kernels.mm(buf520, reinterpret_tensor(primals_267, (24, 24), (1, 24), 0), out=buf522)
    buf523 = empty((1568, 4, 16, 6), device='cpu', dtype=torch.float32)
    buf524 = empty((1568, 4, 6, 16), device='cpu', dtype=torch.float32)
    cpp_fused_clone_111(c_void_p(buf521.data_ptr()), c_void_p(buf523.data_ptr()), c_void_p(buf524.data_ptr()))
    buf525 = empty((6272, 16, 16), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_36], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf523, (6272, 16, 6), (96, 6, 1), 0), reinterpret_tensor(buf524, (6272, 6, 16), (96, 16, 1), 0), out=buf525)
    buf526 = buf472; del buf472  # reuse
    buf527 = reinterpret_tensor(buf525, (1568, 4, 16, 16), (1024, 256, 16, 1), 0); del buf525  # reuse
    buf528 = buf470; del buf470  # reuse
    buf529 = buf527; del buf527  # reuse
    buf530 = empty((1568, 4, 16, 6), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_mul_112(c_void_p(buf529.data_ptr()), c_void_p(buf522.data_ptr()), c_void_p(buf526.data_ptr()), c_void_p(buf528.data_ptr()), c_void_p(buf530.data_ptr()))
    buf531 = reinterpret_tensor(buf522, (6272, 16, 6), (96, 6, 1), 0); del buf522  # reuse
    # Source Nodes: [matmul_37], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf529, (6272, 16, 16), (256, 16, 1), 0), reinterpret_tensor(buf530, (6272, 16, 6), (96, 6, 1), 0), out=buf531)
    buf532 = empty((25088, 24), device='cpu', dtype=torch.float32)
    cpp_fused_view_113(c_void_p(buf531.data_ptr()), c_void_p(buf532.data_ptr()))
    buf533 = reinterpret_tensor(buf531, (25088, 24), (24, 1), 0); del buf531  # reuse
    # Source Nodes: [x_168], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_269, buf532, reinterpret_tensor(primals_268, (24, 24), (1, 24), 0), alpha=1, beta=1, out=buf533)
    del primals_269
    buf534 = buf487; del buf487  # reuse
    buf535 = empty_strided((1568, 16, 1), (16, 1, 25088), device='cpu', dtype=torch.float32)
    buf537 = empty((1568, 16, 24), device='cpu', dtype=torch.float32)
    buf538 = empty((25088, 24), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_114(c_void_p(buf478.data_ptr()), c_void_p(buf486.data_ptr()), c_void_p(buf533.data_ptr()), c_void_p(primals_270.data_ptr()), c_void_p(primals_271.data_ptr()), c_void_p(buf534.data_ptr()), c_void_p(buf535.data_ptr()), c_void_p(buf537.data_ptr()), c_void_p(buf538.data_ptr()))
    del primals_271
    buf539 = empty((25088, 96), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_170], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_273, buf538, reinterpret_tensor(primals_272, (24, 96), (1, 24), 0), alpha=1, beta=1, out=buf539)
    del primals_273
    buf540 = empty((25088, 96), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_115(c_void_p(buf539.data_ptr()), c_void_p(buf540.data_ptr()))
    buf541 = empty((25088, 24), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_174], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_275, buf540, reinterpret_tensor(primals_274, (96, 24), (1, 96), 0), alpha=1, beta=1, out=buf541)
    del primals_275
    buf542 = buf534; del buf534  # reuse
    buf543 = empty_strided((1568, 16, 1), (16, 1, 25088), device='cpu', dtype=torch.float32)
    buf545 = empty((1568, 16, 24), device='cpu', dtype=torch.float32)
    buf575 = empty((25088, 24), device='cpu', dtype=torch.float32)
    buf546 = empty((1568, 384), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_116(c_void_p(buf478.data_ptr()), c_void_p(buf486.data_ptr()), c_void_p(buf533.data_ptr()), c_void_p(buf541.data_ptr()), c_void_p(primals_292.data_ptr()), c_void_p(primals_293.data_ptr()), c_void_p(primals_276.data_ptr()), c_void_p(primals_277.data_ptr()), c_void_p(buf542.data_ptr()), c_void_p(buf543.data_ptr()), c_void_p(buf545.data_ptr()), c_void_p(buf575.data_ptr()), c_void_p(buf546.data_ptr()))
    del primals_277
    del primals_293
    buf547 = empty((1568, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___blocks_9_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_279, buf546, reinterpret_tensor(primals_278, (384, 384), (1, 384), 0), alpha=1, beta=1, out=buf547)
    del primals_279
    buf548 = reinterpret_tensor(buf519, (8, 197, 384), (75648, 384, 1), 0); del buf519  # reuse
    buf549 = reinterpret_tensor(buf512, (8, 197, 1), (197, 1, 1), 0); del buf512  # reuse
    buf550 = empty_strided((8, 197, 1), (197, 1, 1576), device='cpu', dtype=torch.float32)
    buf552 = reinterpret_tensor(buf550, (8, 197, 1), (197, 1, 1), 0); del buf550  # reuse
    buf553 = empty((1576, 384), device='cpu', dtype=torch.float32)
    cpp_fused_cat_native_layer_norm_view_117(c_void_p(buf548.data_ptr()), c_void_p(buf552.data_ptr()), c_void_p(buf493.data_ptr()), c_void_p(buf511.data_ptr()), c_void_p(buf547.data_ptr()), c_void_p(primals_280.data_ptr()), c_void_p(primals_281.data_ptr()), c_void_p(buf549.data_ptr()), c_void_p(buf553.data_ptr()))
    del primals_281
    buf554 = buf499; del buf499  # reuse
    # Source Nodes: [l__mod___blocks_9_attn_out_qk], Original ATen: [aten.mm]
    extern_kernels.mm(buf553, reinterpret_tensor(primals_282, (384, 768), (1, 384), 0), out=buf554)
    buf555 = buf511; del buf511  # reuse
    # Source Nodes: [l__mod___blocks_9_attn_out_v], Original ATen: [aten.mm]
    extern_kernels.mm(buf553, reinterpret_tensor(primals_283, (384, 384), (1, 384), 0), out=buf555)
    buf556 = empty((8, 6, 197, 64), device='cpu', dtype=torch.float32)
    buf557 = empty((8, 6, 64, 197), device='cpu', dtype=torch.float32)
    cpp_fused_clone_118(c_void_p(buf554.data_ptr()), c_void_p(buf556.data_ptr()), c_void_p(buf557.data_ptr()))
    buf558 = empty((48, 197, 197), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_38], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf556, (48, 197, 64), (12608, 64, 1), 0), reinterpret_tensor(buf557, (48, 64, 197), (12608, 197, 1), 0), out=buf558)
    buf559 = buf506; del buf506  # reuse
    buf560 = reinterpret_tensor(buf558, (8, 6, 197, 197), (232854, 38809, 197, 1), 0); del buf558  # reuse
    buf561 = buf504; del buf504  # reuse
    buf562 = buf560; del buf560  # reuse
    buf563 = empty((8, 6, 197, 64), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_mul_119(c_void_p(buf562.data_ptr()), c_void_p(buf555.data_ptr()), c_void_p(buf559.data_ptr()), c_void_p(buf561.data_ptr()), c_void_p(buf563.data_ptr()))
    buf564 = reinterpret_tensor(buf555, (48, 197, 64), (12608, 64, 1), 0); del buf555  # reuse
    # Source Nodes: [matmul_39], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf562, (48, 197, 197), (38809, 197, 1), 0), reinterpret_tensor(buf563, (48, 197, 64), (12608, 64, 1), 0), out=buf564)
    buf565 = empty((1576, 384), device='cpu', dtype=torch.float32)
    cpp_fused_view_120(c_void_p(buf564.data_ptr()), c_void_p(buf565.data_ptr()))
    buf566 = reinterpret_tensor(buf564, (1576, 384), (384, 1), 0); del buf564  # reuse
    # Source Nodes: [x_177], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_285, buf565, reinterpret_tensor(primals_284, (384, 384), (1, 384), 0), alpha=1, beta=1, out=buf566)
    del primals_285
    buf567 = empty_strided((8, 197, 1), (197, 1, 1576), device='cpu', dtype=torch.float32)
    buf568 = empty_strided((8, 197, 1), (197, 1, 1576), device='cpu', dtype=torch.float32)
    buf570 = empty((8, 197, 384), device='cpu', dtype=torch.float32)
    buf571 = empty((1576, 384), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_121(c_void_p(buf548.data_ptr()), c_void_p(buf566.data_ptr()), c_void_p(primals_286.data_ptr()), c_void_p(primals_287.data_ptr()), c_void_p(buf567.data_ptr()), c_void_p(buf568.data_ptr()), c_void_p(buf570.data_ptr()), c_void_p(buf571.data_ptr()))
    del primals_287
    buf572 = empty((1576, 1536), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_179], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_289, buf571, reinterpret_tensor(primals_288, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf572)
    del primals_289
    buf573 = empty((1576, 1536), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_122(c_void_p(buf572.data_ptr()), c_void_p(buf573.data_ptr()))
    buf574 = empty((1576, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_183], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_291, buf573, reinterpret_tensor(primals_290, (1536, 384), (1, 1536), 0), alpha=1, beta=1, out=buf574)
    del primals_291
    buf576 = buf521; del buf521  # reuse
    # Source Nodes: [l__mod___blocks_10_attn_in_qk], Original ATen: [aten.mm]
    extern_kernels.mm(buf575, reinterpret_tensor(primals_294, (24, 48), (1, 24), 0), out=buf576)
    buf577 = reinterpret_tensor(buf547, (25088, 24), (24, 1), 0); del buf547  # reuse
    # Source Nodes: [l__mod___blocks_10_attn_in_v], Original ATen: [aten.mm]
    extern_kernels.mm(buf575, reinterpret_tensor(primals_295, (24, 24), (1, 24), 0), out=buf577)
    buf578 = empty((1568, 4, 16, 6), device='cpu', dtype=torch.float32)
    buf579 = empty((1568, 4, 6, 16), device='cpu', dtype=torch.float32)
    cpp_fused_clone_123(c_void_p(buf576.data_ptr()), c_void_p(buf578.data_ptr()), c_void_p(buf579.data_ptr()))
    buf580 = empty((6272, 16, 16), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_40], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf578, (6272, 16, 6), (96, 6, 1), 0), reinterpret_tensor(buf579, (6272, 6, 16), (96, 16, 1), 0), out=buf580)
    buf581 = buf528; del buf528  # reuse
    buf582 = reinterpret_tensor(buf580, (1568, 4, 16, 16), (1024, 256, 16, 1), 0); del buf580  # reuse
    buf583 = buf526; del buf526  # reuse
    buf584 = buf582; del buf582  # reuse
    buf585 = empty((1568, 4, 16, 6), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_mul_124(c_void_p(buf584.data_ptr()), c_void_p(buf577.data_ptr()), c_void_p(buf581.data_ptr()), c_void_p(buf583.data_ptr()), c_void_p(buf585.data_ptr()))
    buf586 = reinterpret_tensor(buf577, (6272, 16, 6), (96, 6, 1), 0); del buf577  # reuse
    # Source Nodes: [matmul_41], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf584, (6272, 16, 16), (256, 16, 1), 0), reinterpret_tensor(buf585, (6272, 16, 6), (96, 6, 1), 0), out=buf586)
    buf587 = empty((25088, 24), device='cpu', dtype=torch.float32)
    cpp_fused_view_125(c_void_p(buf586.data_ptr()), c_void_p(buf587.data_ptr()))
    buf588 = reinterpret_tensor(buf586, (25088, 24), (24, 1), 0); del buf586  # reuse
    # Source Nodes: [x_186], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_297, buf587, reinterpret_tensor(primals_296, (24, 24), (1, 24), 0), alpha=1, beta=1, out=buf588)
    del primals_297
    buf589 = reinterpret_tensor(buf588, (1568, 16, 24), (384, 24, 1), 0); del buf588  # reuse
    buf590 = buf542; del buf542  # reuse
    buf591 = empty_strided((1568, 16, 1), (16, 1, 25088), device='cpu', dtype=torch.float32)
    buf593 = empty((1568, 16, 24), device='cpu', dtype=torch.float32)
    buf594 = empty((25088, 24), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_126(c_void_p(buf589.data_ptr()), c_void_p(buf478.data_ptr()), c_void_p(buf486.data_ptr()), c_void_p(buf533.data_ptr()), c_void_p(buf541.data_ptr()), c_void_p(primals_298.data_ptr()), c_void_p(primals_299.data_ptr()), c_void_p(buf590.data_ptr()), c_void_p(buf591.data_ptr()), c_void_p(buf593.data_ptr()), c_void_p(buf594.data_ptr()))
    del primals_299
    buf595 = empty((25088, 96), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_188], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_301, buf594, reinterpret_tensor(primals_300, (24, 96), (1, 24), 0), alpha=1, beta=1, out=buf595)
    del primals_301
    buf596 = empty((25088, 96), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_127(c_void_p(buf595.data_ptr()), c_void_p(buf596.data_ptr()))
    buf597 = buf541; del buf541  # reuse
    # Source Nodes: [x_192], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_303, buf596, reinterpret_tensor(primals_302, (96, 24), (1, 96), 0), alpha=1, beta=1, out=buf597)
    del primals_303
    buf598 = buf590; del buf590  # reuse
    buf599 = empty_strided((1568, 16, 1), (16, 1, 25088), device='cpu', dtype=torch.float32)
    buf601 = reinterpret_tensor(buf533, (1568, 16, 24), (384, 24, 1), 0); del buf533  # reuse
    buf631 = buf486; del buf486  # reuse
    buf602 = reinterpret_tensor(buf478, (1568, 384), (384, 1), 0); del buf478  # reuse
    cpp_fused_add_native_layer_norm_view_128(c_void_p(buf589.data_ptr()), c_void_p(buf597.data_ptr()), c_void_p(primals_320.data_ptr()), c_void_p(primals_321.data_ptr()), c_void_p(primals_304.data_ptr()), c_void_p(primals_305.data_ptr()), c_void_p(buf598.data_ptr()), c_void_p(buf599.data_ptr()), c_void_p(buf601.data_ptr()), c_void_p(buf631.data_ptr()), c_void_p(buf602.data_ptr()))
    del primals_305
    del primals_321
    buf603 = empty((1568, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___blocks_10_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_307, buf602, reinterpret_tensor(primals_306, (384, 384), (1, 384), 0), alpha=1, beta=1, out=buf603)
    del primals_307
    buf604 = reinterpret_tensor(buf574, (8, 197, 384), (75648, 384, 1), 0); del buf574  # reuse
    buf605 = reinterpret_tensor(buf567, (8, 197, 1), (197, 1, 1), 0); del buf567  # reuse
    buf606 = empty_strided((8, 197, 1), (197, 1, 1576), device='cpu', dtype=torch.float32)
    buf608 = reinterpret_tensor(buf606, (8, 197, 1), (197, 1, 1), 0); del buf606  # reuse
    buf609 = empty((1576, 384), device='cpu', dtype=torch.float32)
    cpp_fused_cat_native_layer_norm_view_129(c_void_p(buf604.data_ptr()), c_void_p(buf608.data_ptr()), c_void_p(buf548.data_ptr()), c_void_p(buf566.data_ptr()), c_void_p(buf603.data_ptr()), c_void_p(primals_308.data_ptr()), c_void_p(primals_309.data_ptr()), c_void_p(buf605.data_ptr()), c_void_p(buf609.data_ptr()))
    del primals_309
    buf610 = buf554; del buf554  # reuse
    # Source Nodes: [l__mod___blocks_10_attn_out_qk], Original ATen: [aten.mm]
    extern_kernels.mm(buf609, reinterpret_tensor(primals_310, (384, 768), (1, 384), 0), out=buf610)
    buf611 = buf566; del buf566  # reuse
    # Source Nodes: [l__mod___blocks_10_attn_out_v], Original ATen: [aten.mm]
    extern_kernels.mm(buf609, reinterpret_tensor(primals_311, (384, 384), (1, 384), 0), out=buf611)
    buf612 = empty((8, 6, 197, 64), device='cpu', dtype=torch.float32)
    buf613 = empty((8, 6, 64, 197), device='cpu', dtype=torch.float32)
    cpp_fused_clone_130(c_void_p(buf610.data_ptr()), c_void_p(buf612.data_ptr()), c_void_p(buf613.data_ptr()))
    buf614 = empty((48, 197, 197), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_42], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf612, (48, 197, 64), (12608, 64, 1), 0), reinterpret_tensor(buf613, (48, 64, 197), (12608, 197, 1), 0), out=buf614)
    buf615 = buf561; del buf561  # reuse
    buf616 = reinterpret_tensor(buf614, (8, 6, 197, 197), (232854, 38809, 197, 1), 0); del buf614  # reuse
    buf617 = buf559; del buf559  # reuse
    buf618 = buf616; del buf616  # reuse
    buf619 = empty((8, 6, 197, 64), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_mul_131(c_void_p(buf618.data_ptr()), c_void_p(buf611.data_ptr()), c_void_p(buf615.data_ptr()), c_void_p(buf617.data_ptr()), c_void_p(buf619.data_ptr()))
    buf620 = reinterpret_tensor(buf611, (48, 197, 64), (12608, 64, 1), 0); del buf611  # reuse
    # Source Nodes: [matmul_43], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf618, (48, 197, 197), (38809, 197, 1), 0), reinterpret_tensor(buf619, (48, 197, 64), (12608, 64, 1), 0), out=buf620)
    buf621 = empty((1576, 384), device='cpu', dtype=torch.float32)
    cpp_fused_view_132(c_void_p(buf620.data_ptr()), c_void_p(buf621.data_ptr()))
    buf622 = reinterpret_tensor(buf620, (1576, 384), (384, 1), 0); del buf620  # reuse
    # Source Nodes: [x_195], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_313, buf621, reinterpret_tensor(primals_312, (384, 384), (1, 384), 0), alpha=1, beta=1, out=buf622)
    del primals_313
    buf623 = empty_strided((8, 197, 1), (197, 1, 1576), device='cpu', dtype=torch.float32)
    buf624 = empty_strided((8, 197, 1), (197, 1, 1576), device='cpu', dtype=torch.float32)
    buf626 = empty((8, 197, 384), device='cpu', dtype=torch.float32)
    buf627 = empty((1576, 384), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_133(c_void_p(buf604.data_ptr()), c_void_p(buf622.data_ptr()), c_void_p(primals_314.data_ptr()), c_void_p(primals_315.data_ptr()), c_void_p(buf623.data_ptr()), c_void_p(buf624.data_ptr()), c_void_p(buf626.data_ptr()), c_void_p(buf627.data_ptr()))
    del primals_315
    buf628 = empty((1576, 1536), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_197], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_317, buf627, reinterpret_tensor(primals_316, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf628)
    del primals_317
    buf629 = empty((1576, 1536), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_134(c_void_p(buf628.data_ptr()), c_void_p(buf629.data_ptr()))
    buf630 = empty((1576, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_201], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_319, buf629, reinterpret_tensor(primals_318, (1536, 384), (1, 1536), 0), alpha=1, beta=1, out=buf630)
    del primals_319
    buf632 = buf576; del buf576  # reuse
    # Source Nodes: [l__mod___blocks_11_attn_in_qk], Original ATen: [aten.mm]
    extern_kernels.mm(buf631, reinterpret_tensor(primals_322, (24, 48), (1, 24), 0), out=buf632)
    buf633 = reinterpret_tensor(buf603, (25088, 24), (24, 1), 0); del buf603  # reuse
    # Source Nodes: [l__mod___blocks_11_attn_in_v], Original ATen: [aten.mm]
    extern_kernels.mm(buf631, reinterpret_tensor(primals_323, (24, 24), (1, 24), 0), out=buf633)
    buf634 = empty((1568, 4, 16, 6), device='cpu', dtype=torch.float32)
    buf635 = empty((1568, 4, 6, 16), device='cpu', dtype=torch.float32)
    cpp_fused_clone_135(c_void_p(buf632.data_ptr()), c_void_p(buf634.data_ptr()), c_void_p(buf635.data_ptr()))
    del buf632
    buf636 = empty((6272, 16, 16), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_44], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf634, (6272, 16, 6), (96, 6, 1), 0), reinterpret_tensor(buf635, (6272, 6, 16), (96, 16, 1), 0), out=buf636)
    buf637 = buf583; del buf583  # reuse
    buf638 = reinterpret_tensor(buf636, (1568, 4, 16, 16), (1024, 256, 16, 1), 0); del buf636  # reuse
    buf639 = buf581; del buf581  # reuse
    buf640 = buf638; del buf638  # reuse
    buf641 = empty((1568, 4, 16, 6), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_mul_136(c_void_p(buf640.data_ptr()), c_void_p(buf633.data_ptr()), c_void_p(buf637.data_ptr()), c_void_p(buf639.data_ptr()), c_void_p(buf641.data_ptr()))
    del buf637
    del buf639
    buf642 = reinterpret_tensor(buf633, (6272, 16, 6), (96, 6, 1), 0); del buf633  # reuse
    # Source Nodes: [matmul_45], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf640, (6272, 16, 16), (256, 16, 1), 0), reinterpret_tensor(buf641, (6272, 16, 6), (96, 6, 1), 0), out=buf642)
    buf643 = empty((25088, 24), device='cpu', dtype=torch.float32)
    cpp_fused_view_137(c_void_p(buf642.data_ptr()), c_void_p(buf643.data_ptr()))
    buf644 = reinterpret_tensor(buf642, (25088, 24), (24, 1), 0); del buf642  # reuse
    # Source Nodes: [x_204], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_325, buf643, reinterpret_tensor(primals_324, (24, 24), (1, 24), 0), alpha=1, beta=1, out=buf644)
    del primals_325
    buf645 = buf598; del buf598  # reuse
    buf646 = empty_strided((1568, 16, 1), (16, 1, 25088), device='cpu', dtype=torch.float32)
    buf648 = empty((1568, 16, 24), device='cpu', dtype=torch.float32)
    buf649 = empty((25088, 24), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_138(c_void_p(buf589.data_ptr()), c_void_p(buf597.data_ptr()), c_void_p(buf644.data_ptr()), c_void_p(primals_326.data_ptr()), c_void_p(primals_327.data_ptr()), c_void_p(buf645.data_ptr()), c_void_p(buf646.data_ptr()), c_void_p(buf648.data_ptr()), c_void_p(buf649.data_ptr()))
    del primals_327
    buf650 = empty((25088, 96), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_206], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_329, buf649, reinterpret_tensor(primals_328, (24, 96), (1, 24), 0), alpha=1, beta=1, out=buf650)
    del primals_329
    buf651 = empty((25088, 96), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_139(c_void_p(buf650.data_ptr()), c_void_p(buf651.data_ptr()))
    buf652 = empty((25088, 24), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_210], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_331, buf651, reinterpret_tensor(primals_330, (96, 24), (1, 96), 0), alpha=1, beta=1, out=buf652)
    del primals_331
    buf653 = buf645; del buf645  # reuse
    buf654 = empty_strided((1568, 16, 1), (16, 1, 25088), device='cpu', dtype=torch.float32)
    buf656 = empty((1568, 16, 24), device='cpu', dtype=torch.float32)
    buf657 = empty((1568, 384), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_140(c_void_p(buf589.data_ptr()), c_void_p(buf597.data_ptr()), c_void_p(buf644.data_ptr()), c_void_p(buf652.data_ptr()), c_void_p(primals_332.data_ptr()), c_void_p(primals_333.data_ptr()), c_void_p(buf653.data_ptr()), c_void_p(buf654.data_ptr()), c_void_p(buf656.data_ptr()), c_void_p(buf657.data_ptr()))
    del buf589
    del buf597
    del buf644
    del buf653
    del primals_333
    buf658 = reinterpret_tensor(buf652, (1568, 384), (384, 1), 0); del buf652  # reuse
    # Source Nodes: [l__mod___blocks_11_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_335, buf657, reinterpret_tensor(primals_334, (384, 384), (1, 384), 0), alpha=1, beta=1, out=buf658)
    del primals_335
    buf659 = reinterpret_tensor(buf630, (8, 197, 384), (75648, 384, 1), 0); del buf630  # reuse
    buf660 = reinterpret_tensor(buf623, (8, 197, 1), (197, 1, 1), 0); del buf623  # reuse
    buf661 = empty_strided((8, 197, 1), (197, 1, 1576), device='cpu', dtype=torch.float32)
    buf663 = reinterpret_tensor(buf661, (8, 197, 1), (197, 1, 1), 0); del buf661  # reuse
    buf664 = empty((1576, 384), device='cpu', dtype=torch.float32)
    cpp_fused_cat_native_layer_norm_view_141(c_void_p(buf659.data_ptr()), c_void_p(buf663.data_ptr()), c_void_p(buf604.data_ptr()), c_void_p(buf622.data_ptr()), c_void_p(buf658.data_ptr()), c_void_p(primals_336.data_ptr()), c_void_p(primals_337.data_ptr()), c_void_p(buf660.data_ptr()), c_void_p(buf664.data_ptr()))
    del buf658
    del primals_337
    buf665 = buf610; del buf610  # reuse
    # Source Nodes: [l__mod___blocks_11_attn_out_qk], Original ATen: [aten.mm]
    extern_kernels.mm(buf664, reinterpret_tensor(primals_338, (384, 768), (1, 384), 0), out=buf665)
    buf666 = buf622; del buf622  # reuse
    # Source Nodes: [l__mod___blocks_11_attn_out_v], Original ATen: [aten.mm]
    extern_kernels.mm(buf664, reinterpret_tensor(primals_339, (384, 384), (1, 384), 0), out=buf666)
    buf667 = empty((8, 6, 197, 64), device='cpu', dtype=torch.float32)
    buf668 = empty((8, 6, 64, 197), device='cpu', dtype=torch.float32)
    cpp_fused_clone_142(c_void_p(buf665.data_ptr()), c_void_p(buf667.data_ptr()), c_void_p(buf668.data_ptr()))
    del buf665
    buf669 = empty((48, 197, 197), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_46], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf667, (48, 197, 64), (12608, 64, 1), 0), reinterpret_tensor(buf668, (48, 64, 197), (12608, 197, 1), 0), out=buf669)
    buf670 = buf617; del buf617  # reuse
    buf671 = reinterpret_tensor(buf669, (8, 6, 197, 197), (232854, 38809, 197, 1), 0); del buf669  # reuse
    buf672 = buf615; del buf615  # reuse
    buf673 = buf671; del buf671  # reuse
    buf674 = empty((8, 6, 197, 64), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_mul_143(c_void_p(buf673.data_ptr()), c_void_p(buf666.data_ptr()), c_void_p(buf670.data_ptr()), c_void_p(buf672.data_ptr()), c_void_p(buf674.data_ptr()))
    del buf670
    del buf672
    buf675 = reinterpret_tensor(buf666, (48, 197, 64), (12608, 64, 1), 0); del buf666  # reuse
    # Source Nodes: [matmul_47], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf673, (48, 197, 197), (38809, 197, 1), 0), reinterpret_tensor(buf674, (48, 197, 64), (12608, 64, 1), 0), out=buf675)
    buf676 = empty((1576, 384), device='cpu', dtype=torch.float32)
    cpp_fused_view_144(c_void_p(buf675.data_ptr()), c_void_p(buf676.data_ptr()))
    buf677 = reinterpret_tensor(buf675, (1576, 384), (384, 1), 0); del buf675  # reuse
    # Source Nodes: [x_213], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_341, buf676, reinterpret_tensor(primals_340, (384, 384), (1, 384), 0), alpha=1, beta=1, out=buf677)
    del primals_341
    buf678 = empty_strided((8, 197, 1), (197, 1, 1576), device='cpu', dtype=torch.float32)
    buf679 = empty_strided((8, 197, 1), (197, 1, 1576), device='cpu', dtype=torch.float32)
    buf681 = empty((8, 197, 384), device='cpu', dtype=torch.float32)
    buf682 = empty((1576, 384), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_145(c_void_p(buf659.data_ptr()), c_void_p(buf677.data_ptr()), c_void_p(primals_342.data_ptr()), c_void_p(primals_343.data_ptr()), c_void_p(buf678.data_ptr()), c_void_p(buf679.data_ptr()), c_void_p(buf681.data_ptr()), c_void_p(buf682.data_ptr()))
    del primals_343
    buf683 = empty((1576, 1536), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_215], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_345, buf682, reinterpret_tensor(primals_344, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf683)
    del primals_345
    buf684 = empty((1576, 1536), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_146(c_void_p(buf683.data_ptr()), c_void_p(buf684.data_ptr()))
    buf685 = empty((1576, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_219], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_347, buf684, reinterpret_tensor(primals_346, (1536, 384), (1, 1536), 0), alpha=1, beta=1, out=buf685)
    del primals_347
    buf686 = buf678; del buf678  # reuse
    buf687 = empty_strided((8, 197, 1), (197, 1, 1576), device='cpu', dtype=torch.float32)
    buf689 = empty((8, 197, 384), device='cpu', dtype=torch.float32)
    buf690 = empty((8, 384), device='cpu', dtype=torch.float32)
    cpp_fused_add_clone_native_layer_norm_147(c_void_p(buf659.data_ptr()), c_void_p(buf677.data_ptr()), c_void_p(buf685.data_ptr()), c_void_p(primals_348.data_ptr()), c_void_p(primals_349.data_ptr()), c_void_p(buf686.data_ptr()), c_void_p(buf687.data_ptr()), c_void_p(buf689.data_ptr()), c_void_p(buf690.data_ptr()))
    del buf677
    del buf685
    del buf686
    del primals_349
    buf691 = empty((8, 1000), device='cpu', dtype=torch.float32)
    # Source Nodes: [pred], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_351, buf690, reinterpret_tensor(primals_350, (384, 1000), (1, 384), 0), alpha=1, beta=1, out=buf691)
    del primals_351
    buf692 = reinterpret_tensor(buf687, (8, 197, 1), (197, 1, 1), 0); del buf687  # reuse
    buf693 = reinterpret_tensor(buf679, (8, 197, 1), (197, 1, 1), 0); del buf679  # reuse
    buf694 = empty_strided((8, 6, 197, 197), (232854, 1, 1182, 6), device='cpu', dtype=torch.float32)
    buf695 = reinterpret_tensor(buf654, (1568, 16, 1), (16, 1, 1), 0); del buf654  # reuse
    buf696 = reinterpret_tensor(buf646, (1568, 16, 1), (16, 1, 1), 0); del buf646  # reuse
    buf697 = empty_strided((1568, 4, 16, 16), (1024, 1, 64, 4), device='cpu', dtype=torch.float32)
    buf698 = reinterpret_tensor(buf599, (1568, 16, 1), (16, 1, 1), 0); del buf599  # reuse
    buf699 = reinterpret_tensor(buf624, (8, 197, 1), (197, 1, 1), 0); del buf624  # reuse
    buf700 = empty_strided((8, 6, 197, 197), (232854, 1, 1182, 6), device='cpu', dtype=torch.float32)
    buf701 = reinterpret_tensor(buf591, (1568, 16, 1), (16, 1, 1), 0); del buf591  # reuse
    buf702 = empty_strided((1568, 4, 16, 16), (1024, 1, 64, 4), device='cpu', dtype=torch.float32)
    buf703 = reinterpret_tensor(buf543, (1568, 16, 1), (16, 1, 1), 0); del buf543  # reuse
    buf704 = reinterpret_tensor(buf568, (8, 197, 1), (197, 1, 1), 0); del buf568  # reuse
    buf705 = empty_strided((8, 6, 197, 197), (232854, 1, 1182, 6), device='cpu', dtype=torch.float32)
    buf706 = reinterpret_tensor(buf535, (1568, 16, 1), (16, 1, 1), 0); del buf535  # reuse
    buf707 = empty_strided((1568, 4, 16, 16), (1024, 1, 64, 4), device='cpu', dtype=torch.float32)
    buf708 = reinterpret_tensor(buf488, (1568, 16, 1), (16, 1, 1), 0); del buf488  # reuse
    buf709 = reinterpret_tensor(buf513, (8, 197, 1), (197, 1, 1), 0); del buf513  # reuse
    buf710 = empty_strided((8, 6, 197, 197), (232854, 1, 1182, 6), device='cpu', dtype=torch.float32)
    buf711 = reinterpret_tensor(buf480, (1568, 16, 1), (16, 1, 1), 0); del buf480  # reuse
    buf712 = empty_strided((1568, 4, 16, 16), (1024, 1, 64, 4), device='cpu', dtype=torch.float32)
    buf713 = reinterpret_tensor(buf432, (1568, 16, 1), (16, 1, 1), 0); del buf432  # reuse
    buf714 = reinterpret_tensor(buf457, (8, 197, 1), (197, 1, 1), 0); del buf457  # reuse
    buf715 = empty_strided((8, 6, 197, 197), (232854, 1, 1182, 6), device='cpu', dtype=torch.float32)
    buf716 = reinterpret_tensor(buf424, (1568, 16, 1), (16, 1, 1), 0); del buf424  # reuse
    buf717 = empty_strided((1568, 4, 16, 16), (1024, 1, 64, 4), device='cpu', dtype=torch.float32)
    buf718 = reinterpret_tensor(buf377, (1568, 16, 1), (16, 1, 1), 0); del buf377  # reuse
    buf719 = reinterpret_tensor(buf402, (8, 197, 1), (197, 1, 1), 0); del buf402  # reuse
    buf720 = empty_strided((8, 6, 197, 197), (232854, 1, 1182, 6), device='cpu', dtype=torch.float32)
    buf721 = reinterpret_tensor(buf369, (1568, 16, 1), (16, 1, 1), 0); del buf369  # reuse
    buf722 = empty_strided((1568, 4, 16, 16), (1024, 1, 64, 4), device='cpu', dtype=torch.float32)
    buf723 = reinterpret_tensor(buf321, (1568, 16, 1), (16, 1, 1), 0); del buf321  # reuse
    buf724 = reinterpret_tensor(buf346, (8, 197, 1), (197, 1, 1), 0); del buf346  # reuse
    buf725 = empty_strided((8, 6, 197, 197), (232854, 1, 1182, 6), device='cpu', dtype=torch.float32)
    buf726 = reinterpret_tensor(buf313, (1568, 16, 1), (16, 1, 1), 0); del buf313  # reuse
    buf727 = empty_strided((1568, 4, 16, 16), (1024, 1, 64, 4), device='cpu', dtype=torch.float32)
    buf728 = reinterpret_tensor(buf266, (1568, 16, 1), (16, 1, 1), 0); del buf266  # reuse
    buf729 = reinterpret_tensor(buf291, (8, 197, 1), (197, 1, 1), 0); del buf291  # reuse
    buf730 = empty_strided((8, 6, 197, 197), (232854, 1, 1182, 6), device='cpu', dtype=torch.float32)
    buf731 = reinterpret_tensor(buf258, (1568, 16, 1), (16, 1, 1), 0); del buf258  # reuse
    buf732 = empty_strided((1568, 4, 16, 16), (1024, 1, 64, 4), device='cpu', dtype=torch.float32)
    buf733 = reinterpret_tensor(buf210, (1568, 16, 1), (16, 1, 1), 0); del buf210  # reuse
    buf734 = reinterpret_tensor(buf235, (8, 197, 1), (197, 1, 1), 0); del buf235  # reuse
    buf735 = empty_strided((8, 6, 197, 197), (232854, 1, 1182, 6), device='cpu', dtype=torch.float32)
    buf736 = reinterpret_tensor(buf202, (1568, 16, 1), (16, 1, 1), 0); del buf202  # reuse
    buf737 = empty_strided((1568, 4, 16, 16), (1024, 1, 64, 4), device='cpu', dtype=torch.float32)
    buf738 = reinterpret_tensor(buf155, (1568, 16, 1), (16, 1, 1), 0); del buf155  # reuse
    buf739 = reinterpret_tensor(buf180, (8, 197, 1), (197, 1, 1), 0); del buf180  # reuse
    buf740 = empty_strided((8, 6, 197, 197), (232854, 1, 1182, 6), device='cpu', dtype=torch.float32)
    buf741 = reinterpret_tensor(buf147, (1568, 16, 1), (16, 1, 1), 0); del buf147  # reuse
    buf742 = empty_strided((1568, 4, 16, 16), (1024, 1, 64, 4), device='cpu', dtype=torch.float32)
    buf743 = reinterpret_tensor(buf99, (1568, 16, 1), (16, 1, 1), 0); del buf99  # reuse
    buf744 = reinterpret_tensor(buf124, (8, 197, 1), (197, 1, 1), 0); del buf124  # reuse
    buf745 = empty_strided((8, 6, 197, 197), (232854, 1, 1182, 6), device='cpu', dtype=torch.float32)
    buf746 = reinterpret_tensor(buf91, (1568, 16, 1), (16, 1, 1), 0); del buf91  # reuse
    buf747 = empty_strided((1568, 4, 16, 16), (1024, 1, 64, 4), device='cpu', dtype=torch.float32)
    buf748 = reinterpret_tensor(buf44, (1568, 16, 1), (16, 1, 1), 0); del buf44  # reuse
    buf749 = reinterpret_tensor(buf69, (8, 197, 1), (197, 1, 1), 0); del buf69  # reuse
    buf750 = empty_strided((8, 6, 197, 197), (232854, 1, 1182, 6), device='cpu', dtype=torch.float32)
    buf751 = reinterpret_tensor(buf36, (1568, 16, 1), (16, 1, 1), 0); del buf36  # reuse
    buf752 = empty_strided((1568, 4, 16, 16), (1024, 1, 64, 4), device='cpu', dtype=torch.float32)
    cpp_fused_add_detach_native_layer_norm_native_layer_norm_backward_148(c_void_p(buf692.data_ptr()), c_void_p(buf693.data_ptr()), c_void_p(buf695.data_ptr()), c_void_p(buf696.data_ptr()), c_void_p(buf698.data_ptr()), c_void_p(buf699.data_ptr()), c_void_p(buf701.data_ptr()), c_void_p(buf703.data_ptr()), c_void_p(buf704.data_ptr()), c_void_p(buf706.data_ptr()), c_void_p(buf708.data_ptr()), c_void_p(buf709.data_ptr()), c_void_p(buf711.data_ptr()), c_void_p(buf713.data_ptr()), c_void_p(buf714.data_ptr()), c_void_p(buf716.data_ptr()), c_void_p(buf718.data_ptr()), c_void_p(buf719.data_ptr()), c_void_p(buf721.data_ptr()), c_void_p(buf723.data_ptr()), c_void_p(buf724.data_ptr()), c_void_p(buf726.data_ptr()), c_void_p(buf728.data_ptr()), c_void_p(buf729.data_ptr()), c_void_p(buf731.data_ptr()), c_void_p(buf733.data_ptr()), c_void_p(buf734.data_ptr()), c_void_p(buf736.data_ptr()), c_void_p(buf738.data_ptr()), c_void_p(buf739.data_ptr()), c_void_p(buf741.data_ptr()), c_void_p(buf743.data_ptr()), c_void_p(buf744.data_ptr()), c_void_p(buf746.data_ptr()), c_void_p(buf748.data_ptr()), c_void_p(buf749.data_ptr()), c_void_p(buf751.data_ptr()), c_void_p(buf673.data_ptr()), c_void_p(buf640.data_ptr()), c_void_p(buf618.data_ptr()), c_void_p(buf584.data_ptr()), c_void_p(buf562.data_ptr()), c_void_p(buf529.data_ptr()), c_void_p(buf507.data_ptr()), c_void_p(buf473.data_ptr()), c_void_p(buf451.data_ptr()), c_void_p(buf418.data_ptr()), c_void_p(buf396.data_ptr()), c_void_p(buf362.data_ptr()), c_void_p(buf340.data_ptr()), c_void_p(buf307.data_ptr()), c_void_p(buf285.data_ptr()), c_void_p(buf251.data_ptr()), c_void_p(buf229.data_ptr()), c_void_p(buf196.data_ptr()), c_void_p(buf174.data_ptr()), c_void_p(buf140.data_ptr()), c_void_p(buf118.data_ptr()), c_void_p(buf85.data_ptr()), c_void_p(buf63.data_ptr()), c_void_p(buf29.data_ptr()), c_void_p(buf694.data_ptr()), c_void_p(buf697.data_ptr()), c_void_p(buf700.data_ptr()), c_void_p(buf702.data_ptr()), c_void_p(buf705.data_ptr()), c_void_p(buf707.data_ptr()), c_void_p(buf710.data_ptr()), c_void_p(buf712.data_ptr()), c_void_p(buf715.data_ptr()), c_void_p(buf717.data_ptr()), c_void_p(buf720.data_ptr()), c_void_p(buf722.data_ptr()), c_void_p(buf725.data_ptr()), c_void_p(buf727.data_ptr()), c_void_p(buf730.data_ptr()), c_void_p(buf732.data_ptr()), c_void_p(buf735.data_ptr()), c_void_p(buf737.data_ptr()), c_void_p(buf740.data_ptr()), c_void_p(buf742.data_ptr()), c_void_p(buf745.data_ptr()), c_void_p(buf747.data_ptr()), c_void_p(buf750.data_ptr()), c_void_p(buf752.data_ptr()))
    return (buf691, buf0, primals_6, primals_10, primals_12, primals_18, primals_24, primals_28, primals_34, primals_40, primals_46, primals_52, primals_56, primals_62, primals_68, primals_74, primals_80, primals_84, primals_90, primals_96, primals_102, primals_108, primals_112, primals_118, primals_124, primals_130, primals_136, primals_140, primals_146, primals_152, primals_158, primals_164, primals_168, primals_174, primals_180, primals_186, primals_192, primals_196, primals_202, primals_208, primals_214, primals_220, primals_224, primals_230, primals_236, primals_242, primals_248, primals_252, primals_258, primals_264, primals_270, primals_276, primals_280, primals_286, primals_292, primals_298, primals_304, primals_308, primals_314, primals_320, primals_326, primals_332, primals_336, primals_342, primals_348, buf1, buf3, reinterpret_tensor(buf3, (4, 14, 1, 1), (14, 1, 1, 1), 0), buf4, buf5, buf8, buf9, buf10, buf11, buf14, buf16, buf19, buf20, buf32, buf38, buf39, buf40, buf41, buf46, buf47, buf49, buf50, buf53, buf54, buf66, buf71, buf72, buf73, buf74, buf76, buf88, buf93, buf94, buf95, buf96, buf101, buf102, buf104, buf105, buf108, buf109, buf121, buf126, buf127, buf128, buf129, buf131, buf143, buf149, buf150, buf151, buf152, buf157, buf158, buf160, buf161, buf164, buf165, buf177, buf182, buf183, buf184, buf185, buf187, buf199, buf204, buf205, buf206, buf207, buf212, buf213, buf215, buf216, buf219, buf220, buf232, buf237, buf238, buf239, buf240, buf242, buf254, buf260, buf261, buf262, buf263, buf268, buf269, buf271, buf272, buf275, buf276, buf288, buf293, buf294, buf295, buf296, buf298, buf310, buf315, buf316, buf317, buf318, buf323, buf324, buf326, buf327, buf330, buf331, buf343, buf348, buf349, buf350, buf351, buf353, buf365, buf371, buf372, buf373, buf374, buf379, buf380, buf382, buf383, buf386, buf387, buf399, buf404, buf405, buf406, buf407, buf409, buf421, buf426, buf427, buf428, buf429, buf434, buf435, buf437, buf438, buf441, buf442, buf454, buf459, buf460, buf461, buf462, buf464, buf476, buf482, buf483, buf484, buf485, buf490, buf491, buf493, buf494, buf497, buf498, buf510, buf515, buf516, buf517, buf518, buf520, buf532, buf537, buf538, buf539, buf540, buf545, buf546, buf548, buf549, buf552, buf553, buf565, buf570, buf571, buf572, buf573, buf575, buf587, buf593, buf594, buf595, buf596, buf601, buf602, buf604, buf605, buf608, buf609, buf621, buf626, buf627, buf628, buf629, buf631, buf643, buf648, buf649, buf650, buf651, buf656, buf657, buf659, buf660, buf663, buf664, buf676, buf681, buf682, buf683, buf684, buf689, buf690, reinterpret_tensor(primals_350, (1000, 384), (384, 1), 0), buf692, reinterpret_tensor(primals_346, (384, 1536), (1536, 1), 0), reinterpret_tensor(primals_344, (1536, 384), (384, 1), 0), buf693, reinterpret_tensor(primals_340, (384, 384), (384, 1), 0), reinterpret_tensor(buf673, (48, 197, 197), (38809, 1, 197), 0), reinterpret_tensor(buf674, (48, 64, 197), (12608, 1, 64), 0), buf694, reinterpret_tensor(buf667, (48, 64, 197), (12608, 1, 64), 0), reinterpret_tensor(buf668, (48, 197, 64), (12608, 1, 197), 0), reinterpret_tensor(primals_339, (384, 384), (384, 1), 0), reinterpret_tensor(primals_338, (768, 384), (384, 1), 0), reinterpret_tensor(primals_334, (384, 384), (384, 1), 0), buf695, reinterpret_tensor(primals_330, (24, 96), (96, 1), 0), reinterpret_tensor(primals_328, (96, 24), (24, 1), 0), buf696, reinterpret_tensor(primals_324, (24, 24), (24, 1), 0), reinterpret_tensor(buf640, (6272, 16, 16), (256, 1, 16), 0), reinterpret_tensor(buf641, (6272, 6, 16), (96, 1, 6), 0), buf697, reinterpret_tensor(buf634, (6272, 6, 16), (96, 1, 6), 0), reinterpret_tensor(buf635, (6272, 16, 6), (96, 1, 16), 0), reinterpret_tensor(primals_323, (24, 24), (24, 1), 0), reinterpret_tensor(primals_322, (48, 24), (24, 1), 0), buf698, reinterpret_tensor(primals_318, (384, 1536), (1536, 1), 0), reinterpret_tensor(primals_316, (1536, 384), (384, 1), 0), buf699, reinterpret_tensor(primals_312, (384, 384), (384, 1), 0), reinterpret_tensor(buf618, (48, 197, 197), (38809, 1, 197), 0), reinterpret_tensor(buf619, (48, 64, 197), (12608, 1, 64), 0), buf700, reinterpret_tensor(buf612, (48, 64, 197), (12608, 1, 64), 0), reinterpret_tensor(buf613, (48, 197, 64), (12608, 1, 197), 0), reinterpret_tensor(primals_311, (384, 384), (384, 1), 0), reinterpret_tensor(primals_310, (768, 384), (384, 1), 0), reinterpret_tensor(primals_306, (384, 384), (384, 1), 0), reinterpret_tensor(primals_302, (24, 96), (96, 1), 0), reinterpret_tensor(primals_300, (96, 24), (24, 1), 0), buf701, reinterpret_tensor(primals_296, (24, 24), (24, 1), 0), reinterpret_tensor(buf584, (6272, 16, 16), (256, 1, 16), 0), reinterpret_tensor(buf585, (6272, 6, 16), (96, 1, 6), 0), buf702, reinterpret_tensor(buf578, (6272, 6, 16), (96, 1, 6), 0), reinterpret_tensor(buf579, (6272, 16, 6), (96, 1, 16), 0), reinterpret_tensor(primals_295, (24, 24), (24, 1), 0), reinterpret_tensor(primals_294, (48, 24), (24, 1), 0), buf703, reinterpret_tensor(primals_290, (384, 1536), (1536, 1), 0), reinterpret_tensor(primals_288, (1536, 384), (384, 1), 0), buf704, reinterpret_tensor(primals_284, (384, 384), (384, 1), 0), reinterpret_tensor(buf562, (48, 197, 197), (38809, 1, 197), 0), reinterpret_tensor(buf563, (48, 64, 197), (12608, 1, 64), 0), buf705, reinterpret_tensor(buf556, (48, 64, 197), (12608, 1, 64), 0), reinterpret_tensor(buf557, (48, 197, 64), (12608, 1, 197), 0), reinterpret_tensor(primals_283, (384, 384), (384, 1), 0), reinterpret_tensor(primals_282, (768, 384), (384, 1), 0), reinterpret_tensor(primals_278, (384, 384), (384, 1), 0), reinterpret_tensor(primals_274, (24, 96), (96, 1), 0), reinterpret_tensor(primals_272, (96, 24), (24, 1), 0), buf706, reinterpret_tensor(primals_268, (24, 24), (24, 1), 0), reinterpret_tensor(buf529, (6272, 16, 16), (256, 1, 16), 0), reinterpret_tensor(buf530, (6272, 6, 16), (96, 1, 6), 0), buf707, reinterpret_tensor(buf523, (6272, 6, 16), (96, 1, 6), 0), reinterpret_tensor(buf524, (6272, 16, 6), (96, 1, 16), 0), reinterpret_tensor(primals_267, (24, 24), (24, 1), 0), reinterpret_tensor(primals_266, (48, 24), (24, 1), 0), buf708, reinterpret_tensor(primals_262, (384, 1536), (1536, 1), 0), reinterpret_tensor(primals_260, (1536, 384), (384, 1), 0), buf709, reinterpret_tensor(primals_256, (384, 384), (384, 1), 0), reinterpret_tensor(buf507, (48, 197, 197), (38809, 1, 197), 0), reinterpret_tensor(buf508, (48, 64, 197), (12608, 1, 64), 0), buf710, reinterpret_tensor(buf501, (48, 64, 197), (12608, 1, 64), 0), reinterpret_tensor(buf502, (48, 197, 64), (12608, 1, 197), 0), reinterpret_tensor(primals_255, (384, 384), (384, 1), 0), reinterpret_tensor(primals_254, (768, 384), (384, 1), 0), reinterpret_tensor(primals_250, (384, 384), (384, 1), 0), reinterpret_tensor(primals_246, (24, 96), (96, 1), 0), reinterpret_tensor(primals_244, (96, 24), (24, 1), 0), buf711, reinterpret_tensor(primals_240, (24, 24), (24, 1), 0), reinterpret_tensor(buf473, (6272, 16, 16), (256, 1, 16), 0), reinterpret_tensor(buf474, (6272, 6, 16), (96, 1, 6), 0), buf712, reinterpret_tensor(buf467, (6272, 6, 16), (96, 1, 6), 0), reinterpret_tensor(buf468, (6272, 16, 6), (96, 1, 16), 0), reinterpret_tensor(primals_239, (24, 24), (24, 1), 0), reinterpret_tensor(primals_238, (48, 24), (24, 1), 0), buf713, reinterpret_tensor(primals_234, (384, 1536), (1536, 1), 0), reinterpret_tensor(primals_232, (1536, 384), (384, 1), 0), buf714, reinterpret_tensor(primals_228, (384, 384), (384, 1), 0), reinterpret_tensor(buf451, (48, 197, 197), (38809, 1, 197), 0), reinterpret_tensor(buf452, (48, 64, 197), (12608, 1, 64), 0), buf715, reinterpret_tensor(buf445, (48, 64, 197), (12608, 1, 64), 0), reinterpret_tensor(buf446, (48, 197, 64), (12608, 1, 197), 0), reinterpret_tensor(primals_227, (384, 384), (384, 1), 0), reinterpret_tensor(primals_226, (768, 384), (384, 1), 0), reinterpret_tensor(primals_222, (384, 384), (384, 1), 0), reinterpret_tensor(primals_218, (24, 96), (96, 1), 0), reinterpret_tensor(primals_216, (96, 24), (24, 1), 0), buf716, reinterpret_tensor(primals_212, (24, 24), (24, 1), 0), reinterpret_tensor(buf418, (6272, 16, 16), (256, 1, 16), 0), reinterpret_tensor(buf419, (6272, 6, 16), (96, 1, 6), 0), buf717, reinterpret_tensor(buf412, (6272, 6, 16), (96, 1, 6), 0), reinterpret_tensor(buf413, (6272, 16, 6), (96, 1, 16), 0), reinterpret_tensor(primals_211, (24, 24), (24, 1), 0), reinterpret_tensor(primals_210, (48, 24), (24, 1), 0), buf718, reinterpret_tensor(primals_206, (384, 1536), (1536, 1), 0), reinterpret_tensor(primals_204, (1536, 384), (384, 1), 0), buf719, reinterpret_tensor(primals_200, (384, 384), (384, 1), 0), reinterpret_tensor(buf396, (48, 197, 197), (38809, 1, 197), 0), reinterpret_tensor(buf397, (48, 64, 197), (12608, 1, 64), 0), buf720, reinterpret_tensor(buf390, (48, 64, 197), (12608, 1, 64), 0), reinterpret_tensor(buf391, (48, 197, 64), (12608, 1, 197), 0), reinterpret_tensor(primals_199, (384, 384), (384, 1), 0), reinterpret_tensor(primals_198, (768, 384), (384, 1), 0), reinterpret_tensor(primals_194, (384, 384), (384, 1), 0), reinterpret_tensor(primals_190, (24, 96), (96, 1), 0), reinterpret_tensor(primals_188, (96, 24), (24, 1), 0), buf721, reinterpret_tensor(primals_184, (24, 24), (24, 1), 0), reinterpret_tensor(buf362, (6272, 16, 16), (256, 1, 16), 0), reinterpret_tensor(buf363, (6272, 6, 16), (96, 1, 6), 0), buf722, reinterpret_tensor(buf356, (6272, 6, 16), (96, 1, 6), 0), reinterpret_tensor(buf357, (6272, 16, 6), (96, 1, 16), 0), reinterpret_tensor(primals_183, (24, 24), (24, 1), 0), reinterpret_tensor(primals_182, (48, 24), (24, 1), 0), buf723, reinterpret_tensor(primals_178, (384, 1536), (1536, 1), 0), reinterpret_tensor(primals_176, (1536, 384), (384, 1), 0), buf724, reinterpret_tensor(primals_172, (384, 384), (384, 1), 0), reinterpret_tensor(buf340, (48, 197, 197), (38809, 1, 197), 0), reinterpret_tensor(buf341, (48, 64, 197), (12608, 1, 64), 0), buf725, reinterpret_tensor(buf334, (48, 64, 197), (12608, 1, 64), 0), reinterpret_tensor(buf335, (48, 197, 64), (12608, 1, 197), 0), reinterpret_tensor(primals_171, (384, 384), (384, 1), 0), reinterpret_tensor(primals_170, (768, 384), (384, 1), 0), reinterpret_tensor(primals_166, (384, 384), (384, 1), 0), reinterpret_tensor(primals_162, (24, 96), (96, 1), 0), reinterpret_tensor(primals_160, (96, 24), (24, 1), 0), buf726, reinterpret_tensor(primals_156, (24, 24), (24, 1), 0), reinterpret_tensor(buf307, (6272, 16, 16), (256, 1, 16), 0), reinterpret_tensor(buf308, (6272, 6, 16), (96, 1, 6), 0), buf727, reinterpret_tensor(buf301, (6272, 6, 16), (96, 1, 6), 0), reinterpret_tensor(buf302, (6272, 16, 6), (96, 1, 16), 0), reinterpret_tensor(primals_155, (24, 24), (24, 1), 0), reinterpret_tensor(primals_154, (48, 24), (24, 1), 0), buf728, reinterpret_tensor(primals_150, (384, 1536), (1536, 1), 0), reinterpret_tensor(primals_148, (1536, 384), (384, 1), 0), buf729, reinterpret_tensor(primals_144, (384, 384), (384, 1), 0), reinterpret_tensor(buf285, (48, 197, 197), (38809, 1, 197), 0), reinterpret_tensor(buf286, (48, 64, 197), (12608, 1, 64), 0), buf730, reinterpret_tensor(buf279, (48, 64, 197), (12608, 1, 64), 0), reinterpret_tensor(buf280, (48, 197, 64), (12608, 1, 197), 0), reinterpret_tensor(primals_143, (384, 384), (384, 1), 0), reinterpret_tensor(primals_142, (768, 384), (384, 1), 0), reinterpret_tensor(primals_138, (384, 384), (384, 1), 0), reinterpret_tensor(primals_134, (24, 96), (96, 1), 0), reinterpret_tensor(primals_132, (96, 24), (24, 1), 0), buf731, reinterpret_tensor(primals_128, (24, 24), (24, 1), 0), reinterpret_tensor(buf251, (6272, 16, 16), (256, 1, 16), 0), reinterpret_tensor(buf252, (6272, 6, 16), (96, 1, 6), 0), buf732, reinterpret_tensor(buf245, (6272, 6, 16), (96, 1, 6), 0), reinterpret_tensor(buf246, (6272, 16, 6), (96, 1, 16), 0), reinterpret_tensor(primals_127, (24, 24), (24, 1), 0), reinterpret_tensor(primals_126, (48, 24), (24, 1), 0), buf733, reinterpret_tensor(primals_122, (384, 1536), (1536, 1), 0), reinterpret_tensor(primals_120, (1536, 384), (384, 1), 0), buf734, reinterpret_tensor(primals_116, (384, 384), (384, 1), 0), reinterpret_tensor(buf229, (48, 197, 197), (38809, 1, 197), 0), reinterpret_tensor(buf230, (48, 64, 197), (12608, 1, 64), 0), buf735, reinterpret_tensor(buf223, (48, 64, 197), (12608, 1, 64), 0), reinterpret_tensor(buf224, (48, 197, 64), (12608, 1, 197), 0), reinterpret_tensor(primals_115, (384, 384), (384, 1), 0), reinterpret_tensor(primals_114, (768, 384), (384, 1), 0), reinterpret_tensor(primals_110, (384, 384), (384, 1), 0), reinterpret_tensor(primals_106, (24, 96), (96, 1), 0), reinterpret_tensor(primals_104, (96, 24), (24, 1), 0), buf736, reinterpret_tensor(primals_100, (24, 24), (24, 1), 0), reinterpret_tensor(buf196, (6272, 16, 16), (256, 1, 16), 0), reinterpret_tensor(buf197, (6272, 6, 16), (96, 1, 6), 0), buf737, reinterpret_tensor(buf190, (6272, 6, 16), (96, 1, 6), 0), reinterpret_tensor(buf191, (6272, 16, 6), (96, 1, 16), 0), reinterpret_tensor(primals_99, (24, 24), (24, 1), 0), reinterpret_tensor(primals_98, (48, 24), (24, 1), 0), buf738, reinterpret_tensor(primals_94, (384, 1536), (1536, 1), 0), reinterpret_tensor(primals_92, (1536, 384), (384, 1), 0), buf739, reinterpret_tensor(primals_88, (384, 384), (384, 1), 0), reinterpret_tensor(buf174, (48, 197, 197), (38809, 1, 197), 0), reinterpret_tensor(buf175, (48, 64, 197), (12608, 1, 64), 0), buf740, reinterpret_tensor(buf168, (48, 64, 197), (12608, 1, 64), 0), reinterpret_tensor(buf169, (48, 197, 64), (12608, 1, 197), 0), reinterpret_tensor(primals_87, (384, 384), (384, 1), 0), reinterpret_tensor(primals_86, (768, 384), (384, 1), 0), reinterpret_tensor(primals_82, (384, 384), (384, 1), 0), reinterpret_tensor(primals_78, (24, 96), (96, 1), 0), reinterpret_tensor(primals_76, (96, 24), (24, 1), 0), buf741, reinterpret_tensor(primals_72, (24, 24), (24, 1), 0), reinterpret_tensor(buf140, (6272, 16, 16), (256, 1, 16), 0), reinterpret_tensor(buf141, (6272, 6, 16), (96, 1, 6), 0), buf742, reinterpret_tensor(buf134, (6272, 6, 16), (96, 1, 6), 0), reinterpret_tensor(buf135, (6272, 16, 6), (96, 1, 16), 0), reinterpret_tensor(primals_71, (24, 24), (24, 1), 0), reinterpret_tensor(primals_70, (48, 24), (24, 1), 0), buf743, reinterpret_tensor(primals_66, (384, 1536), (1536, 1), 0), reinterpret_tensor(primals_64, (1536, 384), (384, 1), 0), buf744, reinterpret_tensor(primals_60, (384, 384), (384, 1), 0), reinterpret_tensor(buf118, (48, 197, 197), (38809, 1, 197), 0), reinterpret_tensor(buf119, (48, 64, 197), (12608, 1, 64), 0), buf745, reinterpret_tensor(buf112, (48, 64, 197), (12608, 1, 64), 0), reinterpret_tensor(buf113, (48, 197, 64), (12608, 1, 197), 0), reinterpret_tensor(primals_59, (384, 384), (384, 1), 0), reinterpret_tensor(primals_58, (768, 384), (384, 1), 0), reinterpret_tensor(primals_54, (384, 384), (384, 1), 0), reinterpret_tensor(primals_50, (24, 96), (96, 1), 0), reinterpret_tensor(primals_48, (96, 24), (24, 1), 0), buf746, reinterpret_tensor(primals_44, (24, 24), (24, 1), 0), reinterpret_tensor(buf85, (6272, 16, 16), (256, 1, 16), 0), reinterpret_tensor(buf86, (6272, 6, 16), (96, 1, 6), 0), buf747, reinterpret_tensor(buf79, (6272, 6, 16), (96, 1, 6), 0), reinterpret_tensor(buf80, (6272, 16, 6), (96, 1, 16), 0), reinterpret_tensor(primals_43, (24, 24), (24, 1), 0), reinterpret_tensor(primals_42, (48, 24), (24, 1), 0), buf748, reinterpret_tensor(primals_38, (384, 1536), (1536, 1), 0), reinterpret_tensor(primals_36, (1536, 384), (384, 1), 0), buf749, reinterpret_tensor(primals_32, (384, 384), (384, 1), 0), reinterpret_tensor(buf63, (48, 197, 197), (38809, 1, 197), 0), reinterpret_tensor(buf64, (48, 64, 197), (12608, 1, 64), 0), buf750, reinterpret_tensor(buf57, (48, 64, 197), (12608, 1, 64), 0), reinterpret_tensor(buf58, (48, 197, 64), (12608, 1, 197), 0), reinterpret_tensor(primals_31, (384, 384), (384, 1), 0), reinterpret_tensor(primals_30, (768, 384), (384, 1), 0), reinterpret_tensor(primals_26, (384, 384), (384, 1), 0), reinterpret_tensor(primals_22, (24, 96), (96, 1), 0), reinterpret_tensor(primals_20, (96, 24), (24, 1), 0), buf751, reinterpret_tensor(primals_16, (24, 24), (24, 1), 0), reinterpret_tensor(buf29, (6272, 16, 16), (256, 1, 16), 0), reinterpret_tensor(buf30, (6272, 6, 16), (96, 1, 6), 0), buf752, reinterpret_tensor(buf23, (6272, 6, 16), (96, 1, 6), 0), reinterpret_tensor(buf24, (6272, 16, 6), (96, 1, 16), 0), reinterpret_tensor(primals_15, (24, 24), (24, 1), 0), reinterpret_tensor(primals_14, (48, 24), (24, 1), 0), reinterpret_tensor(primals_8, (384, 384), (384, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((1, 24, 4, 4), (384, 16, 4, 1), device='cpu', dtype=torch.float32)
    primals_2 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    primals_3 = rand_strided((1, 197, 384), (75648, 384, 1), device='cpu', dtype=torch.float32)
    primals_4 = rand_strided((24, 3, 7, 7), (147, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_5 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_6 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_7 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_8 = rand_strided((384, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_9 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_10 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_11 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_12 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_13 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_14 = rand_strided((48, 24), (24, 1), device='cpu', dtype=torch.float32)
    primals_15 = rand_strided((24, 24), (24, 1), device='cpu', dtype=torch.float32)
    primals_16 = rand_strided((24, 24), (24, 1), device='cpu', dtype=torch.float32)
    primals_17 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_18 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_19 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_20 = rand_strided((96, 24), (24, 1), device='cpu', dtype=torch.float32)
    primals_21 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    primals_22 = rand_strided((24, 96), (96, 1), device='cpu', dtype=torch.float32)
    primals_23 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_24 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_25 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_26 = rand_strided((384, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_27 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_28 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_29 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_30 = rand_strided((768, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_31 = rand_strided((384, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_32 = rand_strided((384, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_33 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_34 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_35 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_36 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_37 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_38 = rand_strided((384, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_39 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_40 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_41 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_42 = rand_strided((48, 24), (24, 1), device='cpu', dtype=torch.float32)
    primals_43 = rand_strided((24, 24), (24, 1), device='cpu', dtype=torch.float32)
    primals_44 = rand_strided((24, 24), (24, 1), device='cpu', dtype=torch.float32)
    primals_45 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_46 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_47 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_48 = rand_strided((96, 24), (24, 1), device='cpu', dtype=torch.float32)
    primals_49 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    primals_50 = rand_strided((24, 96), (96, 1), device='cpu', dtype=torch.float32)
    primals_51 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_52 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_53 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_54 = rand_strided((384, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_55 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_56 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_57 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_58 = rand_strided((768, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_59 = rand_strided((384, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_60 = rand_strided((384, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_61 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_62 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_63 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_64 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_65 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_66 = rand_strided((384, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_67 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_68 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_69 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_70 = rand_strided((48, 24), (24, 1), device='cpu', dtype=torch.float32)
    primals_71 = rand_strided((24, 24), (24, 1), device='cpu', dtype=torch.float32)
    primals_72 = rand_strided((24, 24), (24, 1), device='cpu', dtype=torch.float32)
    primals_73 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_74 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_75 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_76 = rand_strided((96, 24), (24, 1), device='cpu', dtype=torch.float32)
    primals_77 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    primals_78 = rand_strided((24, 96), (96, 1), device='cpu', dtype=torch.float32)
    primals_79 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_80 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_81 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_82 = rand_strided((384, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_83 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_84 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_85 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_86 = rand_strided((768, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_87 = rand_strided((384, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_88 = rand_strided((384, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_89 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_90 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_91 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_92 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_93 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_94 = rand_strided((384, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_95 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_96 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_97 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_98 = rand_strided((48, 24), (24, 1), device='cpu', dtype=torch.float32)
    primals_99 = rand_strided((24, 24), (24, 1), device='cpu', dtype=torch.float32)
    primals_100 = rand_strided((24, 24), (24, 1), device='cpu', dtype=torch.float32)
    primals_101 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_102 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_103 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_104 = rand_strided((96, 24), (24, 1), device='cpu', dtype=torch.float32)
    primals_105 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    primals_106 = rand_strided((24, 96), (96, 1), device='cpu', dtype=torch.float32)
    primals_107 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_108 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_109 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_110 = rand_strided((384, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_111 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_112 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_113 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_114 = rand_strided((768, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_115 = rand_strided((384, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_116 = rand_strided((384, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_117 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_118 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_119 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_120 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_121 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_122 = rand_strided((384, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_123 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_124 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_125 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_126 = rand_strided((48, 24), (24, 1), device='cpu', dtype=torch.float32)
    primals_127 = rand_strided((24, 24), (24, 1), device='cpu', dtype=torch.float32)
    primals_128 = rand_strided((24, 24), (24, 1), device='cpu', dtype=torch.float32)
    primals_129 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_130 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_131 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_132 = rand_strided((96, 24), (24, 1), device='cpu', dtype=torch.float32)
    primals_133 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    primals_134 = rand_strided((24, 96), (96, 1), device='cpu', dtype=torch.float32)
    primals_135 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_136 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_137 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_138 = rand_strided((384, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_139 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_140 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_141 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_142 = rand_strided((768, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_143 = rand_strided((384, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_144 = rand_strided((384, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_145 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_146 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_147 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_148 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_149 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_150 = rand_strided((384, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_151 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_152 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_153 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_154 = rand_strided((48, 24), (24, 1), device='cpu', dtype=torch.float32)
    primals_155 = rand_strided((24, 24), (24, 1), device='cpu', dtype=torch.float32)
    primals_156 = rand_strided((24, 24), (24, 1), device='cpu', dtype=torch.float32)
    primals_157 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_158 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_159 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_160 = rand_strided((96, 24), (24, 1), device='cpu', dtype=torch.float32)
    primals_161 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    primals_162 = rand_strided((24, 96), (96, 1), device='cpu', dtype=torch.float32)
    primals_163 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_164 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_165 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_166 = rand_strided((384, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_167 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_168 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_169 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_170 = rand_strided((768, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_171 = rand_strided((384, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_172 = rand_strided((384, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_173 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_174 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_175 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_176 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_177 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_178 = rand_strided((384, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_179 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_180 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_181 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_182 = rand_strided((48, 24), (24, 1), device='cpu', dtype=torch.float32)
    primals_183 = rand_strided((24, 24), (24, 1), device='cpu', dtype=torch.float32)
    primals_184 = rand_strided((24, 24), (24, 1), device='cpu', dtype=torch.float32)
    primals_185 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_186 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_187 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_188 = rand_strided((96, 24), (24, 1), device='cpu', dtype=torch.float32)
    primals_189 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    primals_190 = rand_strided((24, 96), (96, 1), device='cpu', dtype=torch.float32)
    primals_191 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_192 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_193 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_194 = rand_strided((384, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_195 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_196 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_197 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_198 = rand_strided((768, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_199 = rand_strided((384, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_200 = rand_strided((384, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_201 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_202 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_203 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_204 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_205 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_206 = rand_strided((384, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_207 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_208 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_209 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_210 = rand_strided((48, 24), (24, 1), device='cpu', dtype=torch.float32)
    primals_211 = rand_strided((24, 24), (24, 1), device='cpu', dtype=torch.float32)
    primals_212 = rand_strided((24, 24), (24, 1), device='cpu', dtype=torch.float32)
    primals_213 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_214 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_215 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_216 = rand_strided((96, 24), (24, 1), device='cpu', dtype=torch.float32)
    primals_217 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    primals_218 = rand_strided((24, 96), (96, 1), device='cpu', dtype=torch.float32)
    primals_219 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_220 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_221 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_222 = rand_strided((384, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_223 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_224 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_225 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_226 = rand_strided((768, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_227 = rand_strided((384, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_228 = rand_strided((384, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_229 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_230 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_231 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_232 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_233 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_234 = rand_strided((384, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_235 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_236 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_237 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_238 = rand_strided((48, 24), (24, 1), device='cpu', dtype=torch.float32)
    primals_239 = rand_strided((24, 24), (24, 1), device='cpu', dtype=torch.float32)
    primals_240 = rand_strided((24, 24), (24, 1), device='cpu', dtype=torch.float32)
    primals_241 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_242 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_243 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_244 = rand_strided((96, 24), (24, 1), device='cpu', dtype=torch.float32)
    primals_245 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    primals_246 = rand_strided((24, 96), (96, 1), device='cpu', dtype=torch.float32)
    primals_247 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_248 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_249 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_250 = rand_strided((384, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_251 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_252 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_253 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_254 = rand_strided((768, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_255 = rand_strided((384, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_256 = rand_strided((384, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_257 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_258 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_259 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_260 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_261 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_262 = rand_strided((384, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_263 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_264 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_265 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_266 = rand_strided((48, 24), (24, 1), device='cpu', dtype=torch.float32)
    primals_267 = rand_strided((24, 24), (24, 1), device='cpu', dtype=torch.float32)
    primals_268 = rand_strided((24, 24), (24, 1), device='cpu', dtype=torch.float32)
    primals_269 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_270 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_271 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_272 = rand_strided((96, 24), (24, 1), device='cpu', dtype=torch.float32)
    primals_273 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    primals_274 = rand_strided((24, 96), (96, 1), device='cpu', dtype=torch.float32)
    primals_275 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_276 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_277 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_278 = rand_strided((384, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_279 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_280 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_281 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_282 = rand_strided((768, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_283 = rand_strided((384, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_284 = rand_strided((384, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_285 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_286 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_287 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_288 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_289 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_290 = rand_strided((384, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_291 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_292 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_293 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_294 = rand_strided((48, 24), (24, 1), device='cpu', dtype=torch.float32)
    primals_295 = rand_strided((24, 24), (24, 1), device='cpu', dtype=torch.float32)
    primals_296 = rand_strided((24, 24), (24, 1), device='cpu', dtype=torch.float32)
    primals_297 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_298 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_299 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_300 = rand_strided((96, 24), (24, 1), device='cpu', dtype=torch.float32)
    primals_301 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    primals_302 = rand_strided((24, 96), (96, 1), device='cpu', dtype=torch.float32)
    primals_303 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_304 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_305 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_306 = rand_strided((384, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_307 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_308 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_309 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_310 = rand_strided((768, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_311 = rand_strided((384, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_312 = rand_strided((384, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_313 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_314 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_315 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_316 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_317 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_318 = rand_strided((384, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_319 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_320 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_321 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_322 = rand_strided((48, 24), (24, 1), device='cpu', dtype=torch.float32)
    primals_323 = rand_strided((24, 24), (24, 1), device='cpu', dtype=torch.float32)
    primals_324 = rand_strided((24, 24), (24, 1), device='cpu', dtype=torch.float32)
    primals_325 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_326 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_327 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_328 = rand_strided((96, 24), (24, 1), device='cpu', dtype=torch.float32)
    primals_329 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    primals_330 = rand_strided((24, 96), (96, 1), device='cpu', dtype=torch.float32)
    primals_331 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_332 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_333 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_334 = rand_strided((384, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_335 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_336 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_337 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_338 = rand_strided((768, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_339 = rand_strided((384, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_340 = rand_strided((384, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_341 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_342 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_343 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_344 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_345 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_346 = rand_strided((384, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    primals_347 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_348 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_349 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_350 = rand_strided((1000, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_351 = rand_strided((1000, ), (1, ), device='cpu', dtype=torch.float32)
    primals_352 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('tnt_s_patch16_224', benchmark_compiled_module)
