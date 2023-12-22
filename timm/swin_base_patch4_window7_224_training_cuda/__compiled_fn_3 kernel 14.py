
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32768, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32', 13: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(12, 13))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_roll_view_13', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr3, out_ptr4, out_ptr5, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r3 = rindex
    x4 = xindex
    x0 = xindex % 56
    x1 = (xindex // 56) % 56
    x2 = (xindex // 3136)
    tmp0 = tl.load(in_ptr0 + (r3 + (128*x4)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r3 + (128*x4)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r3), rmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_ptr3 + (r3 + (128*(((53 + x0) % 56) % 7)) + (896*(((53 + x1) % 56) % 7)) + (6272*(((53 + x0) % 56) // 7)) + (50176*(((53 + x1) % 56) // 7)) + (401408*x2)), rmask & xmask, other=0.0)
    tmp6 = tl.load(in_ptr4 + (r3), rmask, eviction_policy='evict_last', other=0.0)
    tmp8 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp36 = tl.load(in_ptr6 + (r3), rmask, eviction_policy='evict_last', other=0.0)
    tmp38 = tl.load(in_ptr7 + (r3), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp9 = 0.9956521736457944
    tmp10 = tmp8 / tmp9
    tmp11 = tmp7 * tmp10
    tmp12 = tmp4 + tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
    tmp18 = tl.where(rmask & xmask, tmp16, 0)
    tmp19 = tl.sum(tmp18, 1)[:, None]
    tmp20 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tmp19 / tmp21
    tmp23 = tmp13 - tmp22
    tmp24 = tmp23 * tmp23
    tmp25 = tl.broadcast_to(tmp24, [XBLOCK, RBLOCK])
    tmp27 = tl.where(rmask & xmask, tmp25, 0)
    tmp28 = tl.sum(tmp27, 1)[:, None]
    tmp29 = tmp12 - tmp22
    tmp30 = 128.0
    tmp31 = tmp28 / tmp30
    tmp32 = 1e-05
    tmp33 = tmp31 + tmp32
    tmp34 = tl.math.rsqrt(tmp33)
    tmp35 = tmp29 * tmp34
    tmp37 = tmp35 * tmp36
    tmp39 = tmp37 + tmp38
    tmp40 = tmp34 / tmp30
    tl.store(out_ptr0 + (r3 + (128*x4)), tmp12, rmask & xmask)
    tl.store(out_ptr3 + (r3 + (128*x4)), tmp35, rmask & xmask)
    tl.store(out_ptr4 + (r3 + (128*x4)), tmp39, rmask & xmask)
    tl.store(out_ptr5 + (x4), tmp40, xmask)
