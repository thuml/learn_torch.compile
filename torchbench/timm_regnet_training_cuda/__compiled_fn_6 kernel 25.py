
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: 'i32', 18: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(17, 18))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_batch_norm_backward_threshold_backward_24', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'in_out_ptr2']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_out_ptr2, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, out_ptr0, out_ptr1, xnumel, rnumel):
    xnumel = 896
    XBLOCK: tl.constexpr = 1
    rnumel = 784
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex % 196
    r2 = (rindex // 196)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (175616*r2)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (r1 + (196*x0) + (175616*r2)), rmask & xmask, other=0.0)
    tmp6 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (r1 + (196*x0) + (175616*r2)), rmask & xmask, other=0.0)
    tmp16 = tl.load(in_ptr4 + (r1 + (196*x0) + (175616*r2)), rmask & xmask, other=0.0)
    tmp23 = tl.load(in_ptr5 + (r1 + (196*x0) + (175616*r2)), rmask & xmask, other=0.0)
    tmp24 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr7 + (r1 + (196*x0) + (175616*r2)), rmask & xmask, other=0.0)
    tmp32 = tl.load(in_ptr8 + (x0), xmask, eviction_policy='evict_last')
    tmp39 = tl.load(in_ptr9 + (x0), xmask, eviction_policy='evict_last')
    tmp44 = tl.load(in_ptr10 + (x0), xmask, eviction_policy='evict_last')
    tmp48 = tl.load(in_ptr11 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tmp7 = tmp5 - tmp6
    tmp8 = tmp0 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp14 = 0.0
    tmp15 = tmp13 <= tmp14
    tmp17 = tmp0 + tmp16
    tmp18 = tl.where(tmp15, tmp14, tmp17)
    tmp19 = tl.broadcast_to(tmp18, [RBLOCK])
    tmp21 = tl.where(rmask & xmask, tmp19, 0)
    tmp22 = triton_helpers.promote_to_tensor(tl.sum(tmp21, 0))
    tmp25 = tmp23 - tmp24
    tmp26 = tmp18 * tmp25
    tmp27 = tl.broadcast_to(tmp26, [RBLOCK])
    tmp29 = tl.where(rmask & xmask, tmp27, 0)
    tmp30 = triton_helpers.promote_to_tensor(tl.sum(tmp29, 0))
    tmp33 = tmp31 - tmp32
    tmp34 = tmp18 * tmp33
    tmp35 = tl.broadcast_to(tmp34, [RBLOCK])
    tmp37 = tl.where(rmask & xmask, tmp35, 0)
    tmp38 = triton_helpers.promote_to_tensor(tl.sum(tmp37, 0))
    tmp40 = 1e-05
    tmp41 = tmp39 + tmp40
    tmp42 = tl.math.rsqrt(tmp41)
    tmp43 = tmp12 * tmp42
    tmp45 = tmp44 + tmp40
    tmp46 = tl.math.rsqrt(tmp45)
    tmp47 = tmp30 * tmp46
    tmp49 = tmp48 + tmp40
    tmp50 = tl.math.rsqrt(tmp49)
    tmp51 = tmp38 * tmp50
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp43, xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp47, xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr2 + (x0), tmp51, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
    tl.store(out_ptr1 + (x0), tmp22, xmask)
