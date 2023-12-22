
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32768, 64],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_backward_105', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 3136
    x1 = (xindex // 3136)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (3136*r2) + (200704*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (64*x3)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp9 = tl.load(in_ptr3 + (r2 + (64*x3)), rmask & xmask, other=0.0)
    tmp15 = tl.load(in_out_ptr0 + (r2 + (64*x3)), rmask & xmask, other=0.0)
    tmp16 = tl.load(in_ptr4 + (x3), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr6 + (r2 + (64*x3)), rmask & xmask, other=0.0)
    tmp36 = tl.load(in_ptr7 + (x3), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp5, 0)
    tmp8 = tl.sum(tmp7, 1)[:, None]
    tmp10 = tmp4 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
    tmp13 = tl.where(rmask & xmask, tmp11, 0)
    tmp14 = tl.sum(tmp13, 1)[:, None]
    tmp17 = 64.0
    tmp18 = tmp4 * tmp17
    tmp19 = tmp18 - tmp8
    tmp20 = tmp9 * tmp14
    tmp21 = tmp19 - tmp20
    tmp22 = tmp16 * tmp21
    tmp23 = tmp15 + tmp22
    tmp25 = tmp23 * tmp24
    tmp26 = tl.broadcast_to(tmp25, [XBLOCK, RBLOCK])
    tmp28 = tl.where(rmask & xmask, tmp26, 0)
    tmp29 = tl.sum(tmp28, 1)[:, None]
    tmp31 = tmp25 * tmp30
    tmp32 = tl.broadcast_to(tmp31, [XBLOCK, RBLOCK])
    tmp34 = tl.where(rmask & xmask, tmp32, 0)
    tmp35 = tl.sum(tmp34, 1)[:, None]
    tmp37 = tmp25 * tmp17
    tmp38 = tmp37 - tmp29
    tmp39 = tmp30 * tmp35
    tmp40 = tmp38 - tmp39
    tmp41 = tmp36 * tmp40
    tl.store(in_out_ptr0 + (r2 + (64*x3)), tmp23, rmask & xmask)
    tl.store(out_ptr4 + (r2 + (64*x3)), tmp41, rmask & xmask)
