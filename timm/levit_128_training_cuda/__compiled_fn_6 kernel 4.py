
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_3', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 384
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x2), xmask)
    tmp2 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp24 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr8 + (x0), xmask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr9 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 0.125
    tmp6 = tmp4 * tmp5
    tmp8 = 8.0
    tmp9 = tmp7 / tmp8
    tmp10 = 1e-05
    tmp11 = tmp9 + tmp10
    tmp12 = tl.math.rsqrt(tmp11)
    tmp13 = tmp12 * tmp12
    tmp14 = tmp6 * tmp13
    tmp15 = tmp3 * tmp14
    tmp16 = tmp0 - tmp15
    tmp18 = tmp17 * tmp5
    tmp19 = tmp16 - tmp18
    tmp21 = tmp12 * tmp20
    tmp22 = tmp19 * tmp21
    tmp25 = tmp24 * tmp5
    tmp26 = tmp25 * tmp13
    tmp27 = tmp3 * tmp26
    tmp28 = tmp23 - tmp27
    tmp30 = tmp29 * tmp5
    tmp31 = tmp28 - tmp30
    tmp33 = tmp12 * tmp32
    tmp34 = tmp31 * tmp33
    tmp35 = tmp22 + tmp34
    tl.store(in_out_ptr0 + (x2), tmp35, xmask)
