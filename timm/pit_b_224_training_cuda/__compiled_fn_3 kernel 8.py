
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[4096, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_cat_native_layer_norm_view_7', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 2056
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    x0 = xindex % 257
    r2 = rindex
    x1 = (xindex // 257)
    x3 = xindex
    tmp42 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp44 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (r2 + (512*x1)), rmask & tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (tl.broadcast_to(r2, [RBLOCK])), rmask & tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tmp0 >= tmp3
    tmp11 = tl.full([1], 257, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = tl.load(in_ptr2 + ((256*r2) + (131072*x1) + (((-1) + x0) % 256)), rmask & tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp14 = tl.load(in_ptr3 + (tl.broadcast_to(r2, [RBLOCK])), rmask & tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp10, tmp15, tmp16)
    tmp18 = tl.where(tmp4, tmp9, tmp17)
    tmp19 = tl.broadcast_to(tmp18, [RBLOCK])
    tmp21 = tl.where(rmask & xmask, tmp19, 0)
    tmp22 = tl.broadcast_to(tmp19, [RBLOCK])
    tmp24 = tl.where(rmask & xmask, tmp22, 0)
    tmp25 = triton_helpers.promote_to_tensor(tl.sum(tmp24, 0))
    tmp26 = tl.full([1], 512, tl.int32)
    tmp27 = tmp26.to(tl.float32)
    tmp28 = tmp25 / tmp27
    tmp29 = tmp19 - tmp28
    tmp30 = tmp29 * tmp29
    tmp31 = tl.broadcast_to(tmp30, [RBLOCK])
    tmp33 = tl.where(rmask & xmask, tmp31, 0)
    tmp34 = triton_helpers.promote_to_tensor(tl.sum(tmp33, 0))
    tmp35 = 512.0
    tmp36 = tmp34 / tmp35
    tmp37 = 1e-06
    tmp38 = tmp36 + tmp37
    tmp39 = tl.math.rsqrt(tmp38)
    tmp40 = tmp18 - tmp28
    tmp41 = tmp40 * tmp39
    tmp43 = tmp41 * tmp42
    tmp45 = tmp43 + tmp44
    tl.store(out_ptr0 + (r2 + (512*x3)), tmp18, rmask & xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp39, xmask)
    tl.store(out_ptr2 + (r2 + (512*x3)), tmp45, rmask & xmask)
    tl.store(out_ptr1 + (x3), tmp28, xmask)