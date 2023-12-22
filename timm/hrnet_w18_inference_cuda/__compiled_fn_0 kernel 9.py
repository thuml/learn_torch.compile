
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[524288], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(12,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_relu_8', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 451584
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 18
    x5 = (xindex // 56) % 56
    x4 = xindex % 56
    x6 = (xindex // 3136)
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp35 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr7 + (x1), xmask, eviction_policy='evict_last')
    tmp43 = tl.load(in_ptr8 + (x1), xmask, eviction_policy='evict_last')
    tmp45 = tl.load(in_ptr9 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = triton_helpers.maximum(0, tmp16)
    tmp18 = x5
    tmp19 = tmp18.to(tl.float32)
    tmp20 = tmp19 * tmp8
    tmp21 = 0.0
    tmp22 = tmp20 + tmp21
    tmp23 = tmp22 + tmp21
    tmp24 = 0.5
    tmp25 = tmp23 * tmp24
    tmp26 = tmp25.to(tl.int32)
    tmp27 = x4
    tmp28 = tmp27.to(tl.float32)
    tmp29 = tmp28 * tmp8
    tmp30 = tmp29 + tmp21
    tmp31 = tmp30 + tmp21
    tmp32 = tmp31 * tmp24
    tmp33 = tmp32.to(tl.int32)
    tmp34 = tl.load(in_ptr5 + (tmp33 + (28*tmp26) + (784*x6)), xmask, eviction_policy='evict_last')
    tmp36 = tmp34 - tmp35
    tmp38 = tmp37 + tmp4
    tmp39 = tl.sqrt(tmp38)
    tmp40 = 1 / tmp39
    tmp41 = tmp40 * tmp8
    tmp42 = tmp36 * tmp41
    tmp44 = tmp42 * tmp43
    tmp46 = tmp44 + tmp45
    tmp47 = tmp17 + tmp46
    tmp48 = triton_helpers.maximum(0, tmp47)
    tl.store(in_out_ptr0 + (x3), tmp17, xmask)
    tl.store(out_ptr0 + (x3), tmp48, xmask)
