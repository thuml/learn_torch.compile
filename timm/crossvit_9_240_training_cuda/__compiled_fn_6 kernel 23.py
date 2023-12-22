
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_22', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr2, out_ptr3, xnumel, rnumel):
    xnumel = 1576
    XBLOCK: tl.constexpr = 1
    rnumel = 256
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 197
    x1 = (xindex // 197)
    tmp0 = tl.load(in_ptr0 + (r2 + (256*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (256*x3)), rmask & xmask, other=0.0)
    tmp12 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp18 = tl.load(in_ptr4 + (r2 + (256*x3)), rmask & xmask, other=0.0)
    tmp19 = tl.load(in_ptr5 + (x3), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr6 + (x3), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = x0
    tmp4 = tl.full([1], 1, tl.int64)
    tmp5 = tmp3 < tmp4
    tmp6 = tl.load(in_ptr2 + (r2 + (256*x1)), rmask & tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.full(tmp6.shape, 0.0, tmp6.dtype)
    tmp8 = tl.where(tmp5, tmp6, tmp7)
    tmp9 = 0.0
    tmp10 = tl.where(tmp5, tmp8, tmp9)
    tmp11 = tmp2 + tmp10
    tmp13 = tmp11 * tmp12
    tmp14 = tl.broadcast_to(tmp13, [RBLOCK])
    tmp16 = tl.where(rmask & xmask, tmp14, 0)
    tmp17 = triton_helpers.promote_to_tensor(tl.sum(tmp16, 0))
    tmp20 = tmp18 - tmp19
    tmp22 = tmp20 * tmp21
    tmp23 = tmp13 * tmp22
    tmp24 = tl.broadcast_to(tmp23, [RBLOCK])
    tmp26 = tl.where(rmask & xmask, tmp24, 0)
    tmp27 = triton_helpers.promote_to_tensor(tl.sum(tmp26, 0))
    tmp28 = 256.0
    tmp29 = tmp13 * tmp28
    tmp30 = tmp29 - tmp17
    tmp31 = tmp22 * tmp27
    tmp32 = tmp30 - tmp31
    tmp33 = tmp21 / tmp28
    tmp34 = tmp33 * tmp32
    tmp35 = tl.load(in_ptr7 + (r2 + (256*x1)), rmask & tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp36 = tl.full(tmp35.shape, 0.0, tmp35.dtype)
    tmp37 = tl.where(tmp5, tmp35, tmp36)
    tmp38 = tl.where(tmp5, tmp37, tmp9)
    tmp39 = tmp34 + tmp38
    tl.store(out_ptr2 + (r2 + (256*x3)), tmp32, rmask & xmask)
    tl.store(out_ptr3 + (r2 + (256*x3)), tmp39, rmask & xmask)
