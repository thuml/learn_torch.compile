
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


cpp_fused_convolution_backward_sigmoid_backward_0 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    {
        #pragma omp simd simdlen(4) 
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = in_ptr0[static_cast<long>(x0)];
            auto tmp1 = in_ptr1[static_cast<long>(x0)];
            auto tmp2 = static_cast<float>(1.0);
            auto tmp3 = decltype(tmp2)(tmp2 - tmp1);
            auto tmp4 = decltype(tmp1)(tmp1 * tmp3);
            auto tmp5 = decltype(tmp0)(tmp0 * tmp4);
            out_ptr0[static_cast<long>(x0)] = tmp5;
        }
    }
}
''')


cpp_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_1 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0)
{
    auto out_ptr1 = in_out_ptr0;
    auto in_ptr1 = in_out_ptr1;
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (512L*x1)));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (512L*x1)));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 > tmp2);
                    auto tmp5 = static_cast<float>(0.2);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = decltype(tmp4)::blendv(tmp7, tmp4, tmp3);
                    auto tmp11 = tmp9 - tmp10;
                    auto tmp12 = tmp8 * tmp11;
                    tmp_acc0_vec = tmp_acc0_vec + tmp8;
                    tmp_acc1_vec = tmp_acc1_vec + tmp12;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
            auto tmp2 = static_cast<float>(1e-05);
            auto tmp3 = at::vec::Vectorized<float>(tmp2);
            auto tmp4 = tmp1 + tmp3;
            auto tmp5 = tmp4.rsqrt();
            auto tmp6 = tmp0 * tmp5;
            tmp6.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                auto tmp1 = static_cast<float>(0.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = to_float_mask(tmp0 > tmp2);
                auto tmp5 = static_cast<float>(0.2);
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 * tmp6;
                auto tmp8 = decltype(tmp4)::blendv(tmp7, tmp4, tmp3);
                auto tmp10 = static_cast<float>(1e-05);
                auto tmp11 = at::vec::Vectorized<float>(tmp10);
                auto tmp12 = tmp9 + tmp11;
                auto tmp13 = tmp12.rsqrt();
                auto tmp15 = tmp13 * tmp14;
                auto tmp16 = tmp8 * tmp15;
                tmp16.store(in_out_ptr1 + static_cast<long>(x1 + (512L*x0)));
            }
        }
    }
}
''')


cpp_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_2 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0)
{
    auto out_ptr1 = in_out_ptr0;
    auto in_ptr1 = in_out_ptr1;
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (256L*x1)));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (256L*x1)));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 > tmp2);
                    auto tmp5 = static_cast<float>(0.2);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = decltype(tmp4)::blendv(tmp7, tmp4, tmp3);
                    auto tmp11 = tmp9 - tmp10;
                    auto tmp12 = tmp8 * tmp11;
                    tmp_acc0_vec = tmp_acc0_vec + tmp8;
                    tmp_acc1_vec = tmp_acc1_vec + tmp12;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
            auto tmp2 = static_cast<float>(1e-05);
            auto tmp3 = at::vec::Vectorized<float>(tmp2);
            auto tmp4 = tmp1 + tmp3;
            auto tmp5 = tmp4.rsqrt();
            auto tmp6 = tmp0 * tmp5;
            tmp6.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (256L*x0)));
                auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                auto tmp1 = static_cast<float>(0.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = to_float_mask(tmp0 > tmp2);
                auto tmp5 = static_cast<float>(0.2);
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 * tmp6;
                auto tmp8 = decltype(tmp4)::blendv(tmp7, tmp4, tmp3);
                auto tmp10 = static_cast<float>(1e-05);
                auto tmp11 = at::vec::Vectorized<float>(tmp10);
                auto tmp12 = tmp9 + tmp11;
                auto tmp13 = tmp12.rsqrt();
                auto tmp15 = tmp13 * tmp14;
                auto tmp16 = tmp8 * tmp15;
                tmp16.store(in_out_ptr1 + static_cast<long>(x1 + (256L*x0)));
            }
        }
    }
}
''')


cpp_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_3 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0)
{
    auto out_ptr1 = in_out_ptr0;
    auto in_ptr1 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 > tmp2);
                        auto tmp5 = static_cast<float>(0.2);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp8 = decltype(tmp4)::blendv(tmp7, tmp4, tmp3);
                        auto tmp11 = tmp9 - tmp10;
                        auto tmp12 = tmp8 * tmp11;
                        tmp_acc0_vec = tmp_acc0_vec + tmp8;
                        tmp_acc1_vec = tmp_acc1_vec + tmp12;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = static_cast<float>(1e-05);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 + tmp3;
                    auto tmp5 = tmp4.rsqrt();
                    auto tmp6 = tmp0 * tmp5;
                    tmp6.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 > tmp2);
                    auto tmp5 = static_cast<float>(0.2);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = decltype(tmp4)::blendv(tmp7, tmp4, tmp3);
                    auto tmp10 = static_cast<float>(1e-05);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 + tmp11;
                    auto tmp13 = tmp12.rsqrt();
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp16 = tmp8 * tmp15;
                    tmp16.store(in_out_ptr1 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_leaky_relu_backward_4 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(262144L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = to_float_mask(tmp0 > tmp2);
                auto tmp5 = static_cast<float>(0.2);
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 * tmp6;
                auto tmp8 = decltype(tmp4)::blendv(tmp7, tmp4, tmp3);
                tmp8.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_5, primals_6, primals_8, primals_9, primals_11, primals_12, primals_13, primals_15, primals_16, primals_18, primals_19, primals_21, where, convolution_1, where_1, convolution_2, where_2, convolution_3, where_3, sigmoid, tangents_1 = args
    args.clear()
    assert_size_stride(primals_1, (64, 3, 4, 4), (48, 1, 12, 3))
    assert_size_stride(primals_2, (128, 64, 4, 4), (1024, 1, 256, 64))
    assert_size_stride(primals_3, (128, ), (1, ))
    assert_size_stride(primals_5, (256, 128, 4, 4), (2048, 1, 512, 128))
    assert_size_stride(primals_6, (256, ), (1, ))
    assert_size_stride(primals_8, (512, 256, 4, 4), (4096, 1, 1024, 256))
    assert_size_stride(primals_9, (512, ), (1, ))
    assert_size_stride(primals_11, (1, 512, 4, 4), (8192, 1, 2048, 512))
    assert_size_stride(primals_12, (128, ), (1, ))
    assert_size_stride(primals_13, (128, ), (1, ))
    assert_size_stride(primals_15, (256, ), (1, ))
    assert_size_stride(primals_16, (256, ), (1, ))
    assert_size_stride(primals_18, (512, ), (1, ))
    assert_size_stride(primals_19, (512, ), (1, ))
    assert_size_stride(primals_21, (4, 3, 64, 64), (12288, 1, 192, 3))
    assert_size_stride(where, (4, 64, 32, 32), (65536, 1, 2048, 64))
    assert_size_stride(convolution_1, (4, 128, 16, 16), (32768, 1, 2048, 128))
    assert_size_stride(where_1, (4, 128, 16, 16), (32768, 1, 2048, 128))
    assert_size_stride(convolution_2, (4, 256, 8, 8), (16384, 1, 2048, 256))
    assert_size_stride(where_2, (4, 256, 8, 8), (16384, 1, 2048, 256))
    assert_size_stride(convolution_3, (4, 512, 4, 4), (8192, 1, 2048, 512))
    assert_size_stride(where_3, (4, 512, 4, 4), (8192, 1, 2048, 512))
    assert_size_stride(sigmoid, (4, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(tangents_1, (4, 1, 1, 1), (1, 1, 1, 1))
    buf0 = empty((4, 1, 1, 1), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_backward_sigmoid_backward_0(c_void_p(tangents_1.data_ptr()), c_void_p(sigmoid.data_ptr()), c_void_p(buf0.data_ptr()))
    del sigmoid
    del tangents_1
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.sigmoid_backward]
    buf1 = aten.convolution_backward(buf0, where_3, primals_11, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf0
    del primals_11
    buf2 = buf1[0]
    buf3 = buf1[1]
    del buf1
    buf4 = empty((512, ), device='cpu', dtype=torch.float32)
    buf5 = empty((512, ), device='cpu', dtype=torch.float32)
    buf6 = buf5; del buf5  # reuse
    buf7 = buf2; del buf2  # reuse
    cpp_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_1(c_void_p(buf6.data_ptr()), c_void_p(buf7.data_ptr()), c_void_p(where_3.data_ptr()), c_void_p(convolution_3.data_ptr()), c_void_p(primals_18.data_ptr()), c_void_p(primals_19.data_ptr()), c_void_p(primals_9.data_ptr()), c_void_p(buf4.data_ptr()))
    del convolution_3
    del primals_18
    del primals_19
    del primals_9
    del where_3
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
    buf8 = aten.convolution_backward(buf7, where_2, primals_8, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf7
    del primals_8
    buf9 = buf8[0]
    buf10 = buf8[1]
    del buf8
    buf11 = empty((256, ), device='cpu', dtype=torch.float32)
    buf12 = empty((256, ), device='cpu', dtype=torch.float32)
    buf13 = buf12; del buf12  # reuse
    buf14 = buf9; del buf9  # reuse
    cpp_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_2(c_void_p(buf13.data_ptr()), c_void_p(buf14.data_ptr()), c_void_p(where_2.data_ptr()), c_void_p(convolution_2.data_ptr()), c_void_p(primals_15.data_ptr()), c_void_p(primals_16.data_ptr()), c_void_p(primals_6.data_ptr()), c_void_p(buf11.data_ptr()))
    del convolution_2
    del primals_15
    del primals_16
    del primals_6
    del where_2
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
    buf15 = aten.convolution_backward(buf14, where_1, primals_5, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf14
    del primals_5
    buf16 = buf15[0]
    buf17 = buf15[1]
    del buf15
    buf18 = empty((128, ), device='cpu', dtype=torch.float32)
    buf19 = empty((128, ), device='cpu', dtype=torch.float32)
    buf20 = buf19; del buf19  # reuse
    buf21 = buf16; del buf16  # reuse
    cpp_fused_convolution_backward_leaky_relu_backward_native_batch_norm_backward_3(c_void_p(buf20.data_ptr()), c_void_p(buf21.data_ptr()), c_void_p(where_1.data_ptr()), c_void_p(convolution_1.data_ptr()), c_void_p(primals_12.data_ptr()), c_void_p(primals_13.data_ptr()), c_void_p(primals_3.data_ptr()), c_void_p(buf18.data_ptr()))
    del convolution_1
    del primals_12
    del primals_13
    del primals_3
    del where_1
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward, aten.native_batch_norm_backward]
    buf22 = aten.convolution_backward(buf21, where, primals_2, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf21
    del primals_2
    buf23 = buf22[0]
    buf24 = buf22[1]
    del buf22
    buf25 = buf23; del buf23  # reuse
    cpp_fused_convolution_backward_leaky_relu_backward_4(c_void_p(buf25.data_ptr()), c_void_p(where.data_ptr()))
    del where
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.leaky_relu_backward]
    buf26 = aten.convolution_backward(buf25, primals_21, primals_1, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False])
    del buf25
    del primals_1
    del primals_21
    buf27 = buf26[1]
    return (buf27, buf24, buf20, buf18, buf17, buf13, buf11, buf10, buf6, buf4, buf3, None, None, None, None, None, None, None, None, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((64, 3, 4, 4), (48, 1, 12, 3), device='cpu', dtype=torch.float32)
    primals_2 = rand_strided((128, 64, 4, 4), (1024, 1, 256, 64), device='cpu', dtype=torch.float32)
    primals_3 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_5 = rand_strided((256, 128, 4, 4), (2048, 1, 512, 128), device='cpu', dtype=torch.float32)
    primals_6 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_8 = rand_strided((512, 256, 4, 4), (4096, 1, 1024, 256), device='cpu', dtype=torch.float32)
    primals_9 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_11 = rand_strided((1, 512, 4, 4), (8192, 1, 2048, 512), device='cpu', dtype=torch.float32)
    primals_12 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_13 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_15 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_16 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_18 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_19 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_21 = rand_strided((4, 3, 64, 64), (12288, 1, 192, 3), device='cpu', dtype=torch.float32)
    where = rand_strided((4, 64, 32, 32), (65536, 1, 2048, 64), device='cpu', dtype=torch.float32)
    convolution_1 = rand_strided((4, 128, 16, 16), (32768, 1, 2048, 128), device='cpu', dtype=torch.float32)
    where_1 = rand_strided((4, 128, 16, 16), (32768, 1, 2048, 128), device='cpu', dtype=torch.float32)
    convolution_2 = rand_strided((4, 256, 8, 8), (16384, 1, 2048, 256), device='cpu', dtype=torch.float32)
    where_2 = rand_strided((4, 256, 8, 8), (16384, 1, 2048, 256), device='cpu', dtype=torch.float32)
    convolution_3 = rand_strided((4, 512, 4, 4), (8192, 1, 2048, 512), device='cpu', dtype=torch.float32)
    where_3 = rand_strided((4, 512, 4, 4), (8192, 1, 2048, 512), device='cpu', dtype=torch.float32)
    sigmoid = rand_strided((4, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    tangents_1 = rand_strided((4, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_5, primals_6, primals_8, primals_9, primals_11, primals_12, primals_13, primals_15, primals_16, primals_18, primals_19, primals_21, where, convolution_1, where_1, convolution_2, where_2, convolution_3, where_3, sigmoid, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('dcgan', benchmark_compiled_module)
