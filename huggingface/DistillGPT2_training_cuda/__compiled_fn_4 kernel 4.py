
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[33554432], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i64', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_slice_backward_3', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25731584
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 50257)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp8 = tl.load(in_ptr3 + (0))
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK])
    tmp10 = tl.load(in_ptr4 + (0))
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK])
    tmp1 = x1
    tmp2 = tl.full([1], 511, tl.int64)
    tmp3 = tmp1 < tmp2
    tmp4 = tl.load(in_ptr1 + (x2), tmp3 & xmask, other=0.0)
    tmp5 = tl.load(in_ptr2 + (1 + x1), tmp3 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full([1], -100, tl.int64)
    tmp7 = tmp5 != tmp6
    tmp12 = tmp9 / tmp11
    tmp13 = 0.0
    tmp14 = tl.where(tmp7, tmp12, tmp13)
    tmp15 = tmp4 * tmp14
    tmp16 = tl.load(in_ptr5 + (x2), tmp3 & xmask, other=0.0)
    tmp17 = tl.exp(tmp16)
    tmp18 = tl.load(in_ptr6 + (x1), tmp3 & xmask, eviction_policy='evict_last', other=0.0)
    tmp19 = tmp17 * tmp18
    tmp20 = tmp15 - tmp19
    tmp21 = tl.full(tmp20.shape, 0.0, tmp20.dtype)
    tmp22 = tl.where(tmp3, tmp20, tmp21)
    tmp23 = tl.where(tmp3, tmp22, tmp13)
    tmp24 = tmp0 + tmp23
    tl.store(out_ptr0 + (x2), tmp24, xmask)
