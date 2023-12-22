
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_8', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 301056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 768
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_out_ptr0 + (x3), None)
    tmp2 = tl.load(in_ptr1 + (x3), None)
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp4 = 0.002551020408163265
    tmp5 = tmp3 * tmp4
    tmp7 = tmp6 * tmp6
    tmp8 = tmp5 * tmp7
    tmp9 = tmp2 * tmp8
    tmp10 = tmp1 - tmp9
    tmp12 = tmp11 * tmp4
    tmp13 = tmp10 - tmp12
    tmp15 = tmp6 * tmp14
    tmp16 = tmp13 * tmp15
    tmp17 = tmp0 + tmp16
    tl.store(in_out_ptr0 + (x3), tmp17, None)
