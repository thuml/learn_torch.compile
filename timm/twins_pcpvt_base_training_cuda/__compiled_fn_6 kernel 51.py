
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11, 12))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_backward_50', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr3, xnumel, rnumel):
    xnumel = 1568
    XBLOCK: tl.constexpr = 1
    rnumel = 320
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 196
    x1 = (xindex // 196)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (196*r2) + (62720*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (320*x3)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_ptr3 + (r2 + (320*x3)), rmask & xmask, other=0.0)
    tmp11 = tl.load(in_out_ptr0 + (r2 + (320*x3)), rmask & xmask, other=0.0)
    tmp12 = tl.load(in_ptr4 + (x3), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x3), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr6 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp27 = tl.load(in_ptr7 + (r2 + (320*x3)), rmask & xmask, other=0.0)
    tmp33 = tl.load(in_ptr8 + (x3), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 * tmp5
    tmp7 = tl.broadcast_to(tmp6, [RBLOCK])
    tmp9 = tl.where(rmask & xmask, tmp7, 0)
    tmp10 = triton_helpers.promote_to_tensor(tl.sum(tmp9, 0))
    tmp13 = 320.0
    tmp14 = tmp4 * tmp13
    tmp16 = tmp14 - tmp15
    tmp17 = tmp5 * tmp10
    tmp18 = tmp16 - tmp17
    tmp19 = tmp12 * tmp18
    tmp20 = tmp11 + tmp19
    tmp22 = tmp20 * tmp21
    tmp23 = tl.broadcast_to(tmp22, [RBLOCK])
    tmp25 = tl.where(rmask & xmask, tmp23, 0)
    tmp26 = triton_helpers.promote_to_tensor(tl.sum(tmp25, 0))
    tmp28 = tmp22 * tmp27
    tmp29 = tl.broadcast_to(tmp28, [RBLOCK])
    tmp31 = tl.where(rmask & xmask, tmp29, 0)
    tmp32 = triton_helpers.promote_to_tensor(tl.sum(tmp31, 0))
    tmp34 = tmp22 * tmp13
    tmp35 = tmp34 - tmp26
    tmp36 = tmp27 * tmp32
    tmp37 = tmp35 - tmp36
    tmp38 = tmp33 * tmp37
    tl.store(in_out_ptr0 + (r2 + (320*x3)), tmp20, rmask & xmask)
    tl.store(out_ptr3 + (r2 + (320*x3)), tmp38, rmask & xmask)
