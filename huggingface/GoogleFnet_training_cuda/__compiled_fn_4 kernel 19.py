
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i1', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_18', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 512
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + ((2*r1) + (1536*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp11 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (r1 + (768*x0)), rmask & xmask).to(tl.int1)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 * tmp5
    tmp7 = tl.broadcast_to(tmp6, [RBLOCK])
    tmp9 = tl.where(rmask & xmask, tmp7, 0)
    tmp10 = triton_helpers.promote_to_tensor(tl.sum(tmp9, 0))
    tmp12 = 768.0
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 - tmp14
    tmp16 = tmp5 * tmp10
    tmp17 = tmp15 - tmp16
    tmp18 = tmp11 * tmp17
    tmp20 = tmp19.to(tl.float32)
    tmp21 = 1.1111111111111112
    tmp22 = tmp20 * tmp21
    tmp23 = tmp18 * tmp22
    tl.store(out_ptr1 + (r1 + (768*x0)), tmp18, rmask & xmask)
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp23, rmask & xmask)
