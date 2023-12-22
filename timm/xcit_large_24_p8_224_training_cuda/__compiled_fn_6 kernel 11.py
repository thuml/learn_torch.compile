
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mul_native_layer_norm_backward_select_backward_slice_backward_10', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, xnumel, rnumel):
    xnumel = 6280
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 785
    x1 = (xindex // 785)
    tmp0 = tl.load(in_ptr0 + (r2 + (768*x3)), rmask & xmask, other=0.0)
    tmp16 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp22 = tl.load(in_ptr3 + (r2 + (768*x3)), rmask & xmask, other=0.0)
    tmp28 = tl.load(in_ptr4 + (x3), xmask, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = x0
    tmp2 = tl.full([1], 1, tl.int64)
    tmp3 = tmp1 >= tmp2
    tmp4 = tl.load(in_ptr0 + (r2 + (768*x3)), rmask & tmp3 & xmask, other=0.0)
    tmp5 = tl.full(tmp4.shape, 0.0, tmp4.dtype)
    tmp6 = tl.where(tmp3, tmp4, tmp5)
    tmp7 = 0.0
    tmp8 = tl.where(tmp3, tmp6, tmp7)
    tmp9 = tmp0 + tmp8
    tmp10 = tmp1 < tmp2
    tmp11 = tl.load(in_ptr1 + (r2 + (768*x1)), rmask & tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp14 = tl.where(tmp10, tmp13, tmp7)
    tmp15 = tmp9 + tmp14
    tmp17 = tmp15 * tmp16
    tmp18 = tl.broadcast_to(tmp17, [RBLOCK])
    tmp20 = tl.where(rmask & xmask, tmp18, 0)
    tmp21 = triton_helpers.promote_to_tensor(tl.sum(tmp20, 0))
    tmp23 = tmp17 * tmp22
    tmp24 = tl.broadcast_to(tmp23, [RBLOCK])
    tmp26 = tl.where(rmask & xmask, tmp24, 0)
    tmp27 = triton_helpers.promote_to_tensor(tl.sum(tmp26, 0))
    tmp29 = 768.0
    tmp30 = tmp17 * tmp29
    tmp31 = tmp30 - tmp21
    tmp32 = tmp22 * tmp27
    tmp33 = tmp31 - tmp32
    tmp34 = tmp28 * tmp33
    tmp36 = tmp34 * tmp35
    tl.store(out_ptr2 + (r2 + (768*x3)), tmp34, rmask & xmask)
    tl.store(out_ptr3 + (r2 + (768*x3)), tmp36, rmask & xmask)
