
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1048576], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_44', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = (xindex // 75264)
    x1 = (xindex // 196) % 384
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x3), None)
    tmp3 = tl.load(in_ptr2 + (x3), None)
    tmp5 = tl.load(in_ptr3 + (x2), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_out_ptr0 + (x3), None)
    tmp10 = tl.load(in_ptr5 + (x2), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr6 + (x2), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr7 + (x2), None, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp2 = -tmp1
    tmp4 = tmp2 + tmp3
    tmp7 = tmp5 * tmp6
    tmp8 = tmp4 * tmp7
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 - tmp13
    tmp15 = tmp14 * tmp5
    tmp16 = tmp15 * tmp5
    tmp17 = tmp16 * tmp5
    tmp18 = 1.328656462585034e-05
    tmp19 = tmp17 * tmp18
    tmp20 = tmp9 * tmp19
    tmp21 = tmp8 + tmp20
    tmp22 = -tmp19
    tmp23 = tmp22 * tmp11
    tmp24 = tmp10 * tmp5
    tmp25 = tmp24 * tmp18
    tmp26 = tmp23 - tmp25
    tmp27 = tmp21 + tmp26
    tmp28 = tmp0 + tmp27
    tmp30 = tmp28 * tmp29
    tl.store(in_out_ptr0 + (x3), tmp28, None)
    tl.store(out_ptr0 + (x3), tmp30, None)
