
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_backward_select_backward_slice_backward_17', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr2, xnumel, rnumel):
    xnumel = 6280
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    x0 = xindex % 785
    r2 = rindex
    x3 = xindex
    x1 = (xindex // 785)
    tmp10 = tl.load(in_out_ptr0 + (r2 + (768*x3)), rmask & xmask, other=0.0)
    tmp12 = tl.load(in_ptr2 + (r2 + (768*x3)), rmask & xmask, other=0.0)
    tmp16 = tl.load(in_ptr3 + (r2 + (768*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp19 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp25 = tl.load(in_ptr5 + (r2 + (768*x3)), rmask & xmask, other=0.0)
    tmp31 = tl.load(in_ptr0 + (r2 + (768*x3)), rmask & xmask, other=0.0)
    tmp32 = tl.load(in_ptr6 + (x3), xmask, eviction_policy='evict_last')
    tmp0 = x0
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + (r2 + (768*x3)), rmask & tmp2 & xmask, other=0.0)
    tmp4 = tl.load(in_ptr1 + (tl.broadcast_to(r2, [RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tmp3 * tmp4
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp2, tmp5, tmp6)
    tmp8 = 0.0
    tmp9 = tl.where(tmp2, tmp7, tmp8)
    tmp11 = tmp9 + tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = tl.full([1], 0, tl.int32)
    tmp15 = tmp0 == tmp14
    tmp17 = tl.where(tmp15, tmp16, tmp8)
    tmp18 = tmp13 + tmp17
    tmp20 = tmp18 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [RBLOCK])
    tmp23 = tl.where(rmask & xmask, tmp21, 0)
    tmp24 = triton_helpers.promote_to_tensor(tl.sum(tmp23, 0))
    tmp26 = tmp20 * tmp25
    tmp27 = tl.broadcast_to(tmp26, [RBLOCK])
    tmp29 = tl.where(rmask & xmask, tmp27, 0)
    tmp30 = triton_helpers.promote_to_tensor(tl.sum(tmp29, 0))
    tmp33 = 768.0
    tmp34 = tmp20 * tmp33
    tmp35 = tmp34 - tmp24
    tmp36 = tmp25 * tmp30
    tmp37 = tmp35 - tmp36
    tmp38 = tmp32 * tmp37
    tmp39 = tmp31 + tmp38
    tl.store(in_out_ptr0 + (r2 + (768*x3)), tmp18, rmask & xmask)
    tl.store(out_ptr2 + (r2 + (768*x3)), tmp39, rmask & xmask)
