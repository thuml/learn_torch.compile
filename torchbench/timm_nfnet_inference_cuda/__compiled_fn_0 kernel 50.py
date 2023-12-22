
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[262144], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_avg_pool2d_constant_pad_nd_convolution_gelu_mean_mul_relu_sigmoid_49', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 221184
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 36) % 1536
    x4 = (xindex // 36)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x4), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (0))
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK])
    tmp15 = tl.load(in_ptr4 + (x3), None)
    tmp16 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp2 * tmp6
    tmp8 = 2.0
    tmp9 = tmp7 * tmp8
    tmp12 = tmp9 * tmp11
    tmp13 = 0.2
    tmp14 = tmp12 * tmp13
    tmp17 = tmp15 + tmp16
    tmp18 = tmp14 + tmp17
    tmp19 = 0.5
    tmp20 = tmp18 * tmp19
    tmp21 = 0.7071067811865476
    tmp22 = tmp18 * tmp21
    tmp23 = tl.math.erf(tmp22)
    tmp24 = 1.0
    tmp25 = tmp23 + tmp24
    tmp26 = tmp20 * tmp25
    tmp27 = 1.7015043497085571
    tmp28 = tmp26 * tmp27
    tmp29 = 0.9805806756909201
    tmp30 = tmp28 * tmp29
    tl.store(in_out_ptr0 + (x3), tmp18, None)
    tl.store(out_ptr0 + (x3), tmp30, None)
