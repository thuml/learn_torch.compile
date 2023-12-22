
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_batch_norm_backward_19', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 49152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 384
    x1 = (xindex // 384)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (384*(x1 // 16))), None)
    tmp3 = tl.load(in_ptr1 + (x2), None)
    tmp5 = tl.load(in_ptr2 + (x2), None)
    tmp7 = tl.load(in_ptr3 + (x2), None)
    tmp9 = tl.load(in_ptr4 + (x2), None)
    tmp10 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr9 + (x0), None, eviction_policy='evict_last')
    tmp1 = 16.0
    tmp2 = tmp0 / tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp11 = tmp9 - tmp10
    tmp13 = 0.0078125
    tmp14 = tmp12 * tmp13
    tmp16 = tmp15 * tmp15
    tmp17 = tmp14 * tmp16
    tmp18 = tmp11 * tmp17
    tmp19 = tmp8 - tmp18
    tmp21 = tmp20 * tmp13
    tmp22 = tmp19 - tmp21
    tmp24 = tmp15 * tmp23
    tmp25 = tmp22 * tmp24
    tl.store(in_out_ptr0 + (x2), tmp25, None)
