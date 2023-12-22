
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_slice_backward_19', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, rnumel):
    xnumel = 1576
    XBLOCK: tl.constexpr = 1
    rnumel = 384
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    x0 = xindex % 197
    r2 = rindex
    x1 = (xindex // 197)
    x3 = xindex
    tmp8 = tl.load(in_ptr1 + (r2 + (384*x3)), rmask & xmask, other=0.0)
    tmp10 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp16 = tl.load(in_ptr3 + (r2 + (384*x3)), rmask & xmask, other=0.0)
    tmp17 = tl.load(in_ptr4 + (x3), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr5 + (x3), xmask, eviction_policy='evict_last')
    tmp0 = x0
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.load(in_ptr0 + (r2 + (384*x1)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.full(tmp3.shape, 0.0, tmp3.dtype)
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp6 = 0.0
    tmp7 = tl.where(tmp2, tmp5, tmp6)
    tmp9 = tmp7 + tmp8
    tmp11 = tmp9 * tmp10
    tmp12 = tl.broadcast_to(tmp11, [RBLOCK])
    tmp14 = tl.where(rmask & xmask, tmp12, 0)
    tmp15 = triton_helpers.promote_to_tensor(tl.sum(tmp14, 0))
    tmp18 = tmp16 - tmp17
    tmp20 = tmp18 * tmp19
    tmp21 = tmp11 * tmp20
    tmp22 = tl.broadcast_to(tmp21, [RBLOCK])
    tmp24 = tl.where(rmask & xmask, tmp22, 0)
    tmp25 = triton_helpers.promote_to_tensor(tl.sum(tmp24, 0))
    tmp26 = tmp0 >= tmp1
    tmp27 = tl.load(in_out_ptr0 + (r2 + (384*x3)), rmask & tmp26 & xmask, other=0.0)
    tmp28 = tl.broadcast_to(x0, [RBLOCK])
    tmp29 = tmp28 < tmp1
    tmp30 = tmp29 & tmp26
    tmp31 = tl.load(in_ptr6 + (r2 + (384*x1)), rmask & tmp30 & xmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.full(tmp31.shape, 0.0, tmp31.dtype)
    tmp33 = tl.where(tmp30, tmp31, tmp32)
    tmp34 = tl.where(tmp29, tmp33, tmp6)
    tmp35 = tmp27 + tmp34
    tmp36 = tl.full(tmp35.shape, 0.0, tmp35.dtype)
    tmp37 = tl.where(tmp26, tmp35, tmp36)
    tmp38 = tl.where(tmp26, tmp37, tmp6)
    tmp39 = 384.0
    tmp40 = tmp19 / tmp39
    tmp41 = tmp11 * tmp39
    tmp42 = tmp41 - tmp15
    tmp43 = tmp20 * tmp25
    tmp44 = tmp42 - tmp43
    tmp45 = tmp40 * tmp44
    tmp46 = tmp38 + tmp45
    tl.store(in_out_ptr0 + (r2 + (384*x3)), tmp46, rmask & xmask)
