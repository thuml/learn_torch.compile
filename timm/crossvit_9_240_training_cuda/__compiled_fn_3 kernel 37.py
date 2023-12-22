
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[4096, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32', 13: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(12, 13))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_cat_native_layer_norm_36', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 3208
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    x0 = xindex % 401
    r2 = rindex
    x1 = (xindex // 401)
    x3 = xindex
    tmp59 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp61 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp0 = x0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (r2 + (128*x1)), rmask & tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 401, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + (r2 + (128*x3)), rmask & tmp8 & xmask, other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp8, tmp11, tmp12)
    tmp14 = tl.where(tmp4, tmp7, tmp13)
    tmp15 = tl.load(in_ptr2 + (r2 + (128*x1)), rmask & tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp4, tmp15, tmp16)
    tmp18 = tl.where(tmp4, tmp17, tmp13)
    tmp19 = tl.broadcast_to(tmp18, [XBLOCK, RBLOCK])
    tmp21 = tl.where(rmask & xmask, tmp19, 0)
    tmp22 = tl.broadcast_to(tmp19, [XBLOCK, RBLOCK])
    tmp24 = tl.where(rmask & xmask, tmp22, 0)
    tmp25 = tl.sum(tmp24, 1)[:, None]
    tmp26 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp27 = tmp26.to(tl.float32)
    tmp28 = tmp25 / tmp27
    tmp29 = tmp19 - tmp28
    tmp30 = tmp29 * tmp29
    tmp31 = tl.broadcast_to(tmp30, [XBLOCK, RBLOCK])
    tmp33 = tl.where(rmask & xmask, tmp31, 0)
    tmp34 = tl.sum(tmp33, 1)[:, None]
    tmp35 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
    tmp37 = tl.where(rmask & xmask, tmp35, 0)
    tmp38 = tl.broadcast_to(tmp35, [XBLOCK, RBLOCK])
    tmp40 = tl.where(rmask & xmask, tmp38, 0)
    tmp41 = tl.sum(tmp40, 1)[:, None]
    tmp42 = tmp41 / tmp27
    tmp43 = tmp35 - tmp42
    tmp44 = tmp43 * tmp43
    tmp45 = tl.broadcast_to(tmp44, [XBLOCK, RBLOCK])
    tmp47 = tl.where(rmask & xmask, tmp45, 0)
    tmp48 = tl.sum(tmp47, 1)[:, None]
    tmp49 = 128.0
    tmp50 = tmp34 / tmp49
    tmp51 = 1e-06
    tmp52 = tmp50 + tmp51
    tmp53 = tl.math.rsqrt(tmp52)
    tmp54 = tmp48 / tmp49
    tmp55 = tmp54 + tmp51
    tmp56 = tl.math.rsqrt(tmp55)
    tmp57 = tmp18 - tmp28
    tmp58 = tmp57 * tmp53
    tmp60 = tmp58 * tmp59
    tmp62 = tmp60 + tmp61
    tl.store(out_ptr0 + (r2 + (128*x3)), tmp14, rmask & xmask)
    tl.store(out_ptr1 + (r2 + (128*x3)), tmp18, rmask & xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp53, xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x3), tmp56, xmask)
    tl.store(out_ptr4 + (r2 + (128*x3)), tmp62, rmask & xmask)
    tl.store(out_ptr2 + (x3), tmp28, xmask)
    tl.store(out_ptr3 + (x3), tmp42, xmask)
