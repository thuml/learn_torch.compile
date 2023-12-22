
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32768, 32],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: 'i32', 14: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(13, 14))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_39', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 24
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    x3 = (xindex // 16)
    x2 = xindex % 16
    tmp0 = tl.load(in_ptr0 + (r1 + (24*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (24*x0)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp9 = tl.load(in_ptr3 + (r1 + (24*x0)), rmask & xmask, other=0.0)
    tmp10 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_out_ptr0 + (r1 + (24*x0)), rmask & xmask, other=0.0)
    tmp28 = tl.load(in_ptr6 + (x3), xmask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr7 + (r1 + (24*x0)), rmask & xmask, other=0.0)
    tmp32 = tl.load(in_ptr8 + (r1 + (24*x2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp35 = tl.load(in_ptr9 + (x3), xmask, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr10 + (x3), xmask, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr11 + (x3), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp5, 0)
    tmp8 = tl.sum(tmp7, 1)[:, None]
    tmp11 = tmp9 - tmp10
    tmp13 = tmp11 * tmp12
    tmp14 = tmp4 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
    tmp17 = tl.where(rmask & xmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None]
    tmp20 = 24.0
    tmp21 = tmp12 / tmp20
    tmp22 = tmp4 * tmp20
    tmp23 = tmp22 - tmp8
    tmp24 = tmp13 * tmp18
    tmp25 = tmp23 - tmp24
    tmp26 = tmp21 * tmp25
    tmp27 = tmp19 + tmp26
    tmp29 = 384.0
    tmp30 = tmp28 / tmp29
    tmp33 = tmp31 * tmp32
    tmp34 = tmp33 * tmp29
    tmp36 = tmp34 - tmp35
    tmp38 = tmp9 - tmp37
    tmp39 = tmp38 * tmp28
    tmp41 = tmp39 * tmp40
    tmp42 = tmp36 - tmp41
    tmp43 = tmp30 * tmp42
    tmp44 = tmp27 + tmp43
    tl.store(in_out_ptr0 + (r1 + (24*x0)), tmp44, rmask & xmask)
