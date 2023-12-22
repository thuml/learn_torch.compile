
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[131072], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_div_hardswish_backward_native_batch_norm_backward_6', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 256
    x2 = (xindex // 12544)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp5 = tl.load(in_ptr1 + (x0 + (256*x2)), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr2 + (x3), None)
    tmp16 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp1 = -3.0
    tmp2 = tmp0 < tmp1
    tmp3 = 3.0
    tmp4 = tmp0 <= tmp3
    tmp6 = 49.0
    tmp7 = tmp5 / tmp6
    tmp8 = tmp0 / tmp3
    tmp9 = 0.5
    tmp10 = tmp8 + tmp9
    tmp11 = tmp7 * tmp10
    tmp12 = tl.where(tmp4, tmp11, tmp7)
    tmp13 = 0.0
    tmp14 = tl.where(tmp2, tmp13, tmp12)
    tmp17 = tmp15 - tmp16
    tmp19 = 0.002551020408163265
    tmp20 = tmp18 * tmp19
    tmp22 = tmp21 * tmp21
    tmp23 = tmp20 * tmp22
    tmp24 = tmp17 * tmp23
    tmp25 = tmp14 - tmp24
    tmp27 = tmp26 * tmp19
    tmp28 = tmp25 - tmp27
    tmp30 = tmp21 * tmp29
    tmp31 = tmp28 * tmp30
    tl.store(out_ptr0 + (x3), tmp31, None)
