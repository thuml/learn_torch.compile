
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(12,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_threshold_backward_137', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 128
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_out_ptr0 + (x3), None)
    tmp5 = tl.load(in_ptr1 + (x3), None)
    tmp6 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr6 + (x3), None)
    tmp22 = tl.load(in_ptr7 + (x3), None)
    tmp24 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr9 + (x1), None, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr10 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = 3.985969387755102e-05
    tmp8 = tmp6 * tmp7
    tmp10 = tmp9 * tmp9
    tmp11 = tmp8 * tmp10
    tmp12 = tmp5 * tmp11
    tmp13 = tmp4 - tmp12
    tmp15 = tmp14 * tmp7
    tmp16 = tmp13 - tmp15
    tmp18 = tmp9 * tmp17
    tmp19 = tmp16 * tmp18
    tmp21 = tmp20 <= tmp1
    tmp23 = tl.where(tmp21, tmp1, tmp22)
    tmp25 = tmp24 * tmp7
    tmp26 = tmp25 * tmp10
    tmp27 = tmp5 * tmp26
    tmp28 = tmp23 - tmp27
    tmp30 = tmp29 * tmp7
    tmp31 = tmp28 - tmp30
    tmp33 = tmp9 * tmp32
    tmp34 = tmp31 * tmp33
    tmp35 = tmp19 + tmp34
    tl.store(in_out_ptr0 + (x3), tmp35, None)
