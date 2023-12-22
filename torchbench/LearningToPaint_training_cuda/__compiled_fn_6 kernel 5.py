
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 64],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: '*fp32', 18: 'i32', 19: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(18, 19))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_avg_pool2d_backward_native_batch_norm_backward_threshold_backward_4', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'in_out_ptr2']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_out_ptr2, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r3 = rindex
    x0 = xindex
    r2 = (rindex // 16)
    r1 = rindex % 16
    tmp0 = tl.load(in_ptr0 + (x0 + (512*r3)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (x0 + (512*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tl.load(in_ptr2 + (x0 + (512*r3)), rmask & xmask, other=0.0)
    tmp16 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr4 + (x0 + (512*r3)), rmask & xmask, other=0.0)
    tmp25 = tl.load(in_ptr5 + (r1 + (16*x0) + (8192*r2)), rmask & xmask, other=0.0)
    tmp32 = tl.load(in_ptr6 + (x0 + (512*r3)), rmask & xmask, other=0.0)
    tmp33 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr8 + (x0 + (512*r3)), rmask & xmask, other=0.0)
    tmp41 = tl.load(in_ptr9 + (x0), xmask, eviction_policy='evict_last')
    tmp48 = tl.load(in_ptr10 + (x0), xmask, eviction_policy='evict_last')
    tmp53 = tl.load(in_ptr11 + (x0), xmask, eviction_policy='evict_last')
    tmp57 = tl.load(in_ptr12 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tmp3 / 16
    tmp5 = tl.full([1, 1], 0, tl.int32)
    tmp6 = tl.full([1, 1], 1, tl.int32)
    tmp7 = tmp5 < tmp6
    tmp8 = tmp7 & tmp7
    tmp9 = tl.where(tmp8, tmp4, tmp1)
    tmp10 = tl.where(tmp2, tmp1, tmp9)
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
    tmp13 = tl.where(rmask & xmask, tmp11, 0)
    tmp14 = tl.sum(tmp13, 1)[:, None]
    tmp17 = tmp15 - tmp16
    tmp18 = tmp10 * tmp17
    tmp19 = tl.broadcast_to(tmp18, [XBLOCK, RBLOCK])
    tmp21 = tl.where(rmask & xmask, tmp19, 0)
    tmp22 = tl.sum(tmp21, 1)[:, None]
    tmp24 = tmp23 <= tmp1
    tmp26 = tmp10 + tmp25
    tmp27 = tl.where(tmp24, tmp1, tmp26)
    tmp28 = tl.broadcast_to(tmp27, [XBLOCK, RBLOCK])
    tmp30 = tl.where(rmask & xmask, tmp28, 0)
    tmp31 = tl.sum(tmp30, 1)[:, None]
    tmp34 = tmp32 - tmp33
    tmp35 = tmp27 * tmp34
    tmp36 = tl.broadcast_to(tmp35, [XBLOCK, RBLOCK])
    tmp38 = tl.where(rmask & xmask, tmp36, 0)
    tmp39 = tl.sum(tmp38, 1)[:, None]
    tmp42 = tmp40 - tmp41
    tmp43 = tmp27 * tmp42
    tmp44 = tl.broadcast_to(tmp43, [XBLOCK, RBLOCK])
    tmp46 = tl.where(rmask & xmask, tmp44, 0)
    tmp47 = tl.sum(tmp46, 1)[:, None]
    tmp49 = 1e-05
    tmp50 = tmp48 + tmp49
    tmp51 = tl.math.rsqrt(tmp50)
    tmp52 = tmp22 * tmp51
    tmp54 = tmp53 + tmp49
    tmp55 = tl.math.rsqrt(tmp54)
    tmp56 = tmp39 * tmp55
    tmp58 = tmp57 + tmp49
    tmp59 = tl.math.rsqrt(tmp58)
    tmp60 = tmp47 * tmp59
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp52, xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp56, xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr2 + (x0), tmp60, xmask)
    tl.store(out_ptr0 + (x0), tmp14, xmask)
    tl.store(out_ptr1 + (x0), tmp31, xmask)
