
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32', 13: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(12, 13))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_avg_pool2d_backward_native_batch_norm_backward_threshold_backward_14', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, rnumel):
    xnumel = 1024
    XBLOCK: tl.constexpr = 1
    rnumel = 784
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r3 = (rindex // 196)
    r4 = rindex % 196
    x0 = xindex
    r1 = rindex % 14
    r2 = (rindex // 14) % 14
    tmp0 = tl.load(in_ptr0 + (r4 + (196*x0) + (200704*r3)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr1 + ((7*(tl.math.min(tl.math.max(0, (r2 // 2)), (-1) + (tl.math.min(7, 1 + (r2 // 2)))))) + (7*(tl.where((tl.math.min(tl.math.max(0, (r2 // 2)), (-1) + (tl.math.min(7, 1 + (r2 // 2))))) >= 0, 0, 7))) + (49*x0) + (50176*r3) + (tl.math.min(tl.math.max(0, (r1 // 2)), (-1) + (tl.math.min(7, 1 + (r1 // 2))))) + (tl.where((tl.math.min(tl.math.max(0, (r1 // 2)), (-1) + (tl.math.min(7, 1 + (r1 // 2))))) >= 0, 0, 7))), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tl.load(in_ptr2 + (r4 + (196*x0) + (200704*r3)), rmask & xmask, other=0.0)
    tmp20 = tl.load(in_ptr3 + (r4 + (196*x0) + (200704*r3)), rmask & xmask, other=0.0)
    tmp21 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr5 + (r4 + (196*x0) + (200704*r3)), rmask & xmask, other=0.0)
    tmp29 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp36 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr8 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tmp3 / 4
    tmp5 = tl.math.max(0, (r2 // 2))
    tmp6 = tl.math.min(7, 1 + (r2 // 2))
    tmp7 = tmp5 < tmp6
    tmp8 = tl.math.max(0, (r1 // 2))
    tmp9 = tl.math.min(7, 1 + (r1 // 2))
    tmp10 = tmp8 < tmp9
    tmp11 = tmp7 & tmp10
    tmp12 = tl.where(tmp11, tmp4, tmp1)
    tmp14 = tmp12 + tmp13
    tmp15 = tl.where(tmp2, tmp1, tmp14)
    tmp16 = tl.broadcast_to(tmp15, [RBLOCK])
    tmp18 = tl.where(rmask & xmask, tmp16, 0)
    tmp19 = triton_helpers.promote_to_tensor(tl.sum(tmp18, 0))
    tmp22 = tmp20 - tmp21
    tmp23 = tmp15 * tmp22
    tmp24 = tl.broadcast_to(tmp23, [RBLOCK])
    tmp26 = tl.where(rmask & xmask, tmp24, 0)
    tmp27 = triton_helpers.promote_to_tensor(tl.sum(tmp26, 0))
    tmp30 = tmp28 - tmp29
    tmp31 = tmp15 * tmp30
    tmp32 = tl.broadcast_to(tmp31, [RBLOCK])
    tmp34 = tl.where(rmask & xmask, tmp32, 0)
    tmp35 = triton_helpers.promote_to_tensor(tl.sum(tmp34, 0))
    tmp37 = 1e-05
    tmp38 = tmp36 + tmp37
    tmp39 = tl.math.rsqrt(tmp38)
    tmp40 = tmp27 * tmp39
    tmp42 = tmp41 + tmp37
    tmp43 = tl.math.rsqrt(tmp42)
    tmp44 = tmp35 * tmp43
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp40, xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp44, xmask)
    tl.store(out_ptr0 + (x0), tmp19, xmask)
