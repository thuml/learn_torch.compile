
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_fill_mul_sigmoid_silu_sub_28', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x4 = (xindex // 784)
    x1 = (xindex // 784) % 512
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x4), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr2 + (x3), None)
    tmp9 = tl.load(in_ptr3 + (x4), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x3), None)
    tmp15 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tmp4 = 2.0
    tmp5 = tmp3 * tmp4
    tmp6 = 0.2
    tmp7 = tmp5 * tmp6
    tmp10 = tl.sigmoid(tmp9)
    tmp11 = tmp8 * tmp10
    tmp12 = tmp11 * tmp4
    tmp13 = tmp12 * tmp6
    tmp16 = tmp14 + tmp15
    tmp17 = tmp13 + tmp16
    tmp18 = tmp7 + tmp17
    tmp19 = tl.sigmoid(tmp18)
    tmp20 = tmp18 * tmp19
    tmp21 = 0.9622504486493761
    tmp22 = tmp20 * tmp21
    tmp23 = 1.0
    tmp24 = tmp23 - tmp19
    tmp25 = tmp18 * tmp24
    tmp26 = tmp25 + tmp23
    tmp27 = tmp19 * tmp26
    tl.store(out_ptr1 + (x3), tmp22, None)
    tl.store(out_ptr2 + (x3), tmp27, None)
