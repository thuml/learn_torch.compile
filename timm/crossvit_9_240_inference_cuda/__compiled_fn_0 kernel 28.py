
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_gelu_native_layer_norm_27', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 8
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp18 = tl.load(in_out_ptr0 + (r1 + (128*x0)), rmask & xmask, other=0.0)
    tmp19 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp45 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp47 = tl.load(in_ptr6 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp0 = tl.full([1, 1], 0, tl.int64)
    tmp1 = tmp0 >= tmp0
    tmp2 = tl.full([1, 1], 1, tl.int64)
    tmp3 = tmp0 < tmp2
    tmp4 = tl.load(in_ptr0 + (r1 + (128*x0)), rmask & tmp3 & xmask, other=0.0)
    tmp5 = tl.full(tmp4.shape, 0.0, tmp4.dtype)
    tmp6 = tl.where(tmp3, tmp4, tmp5)
    tmp7 = tmp0 >= tmp2
    tmp8 = tl.full([1, 1], 401, tl.int64)
    tmp9 = tmp0 < tmp8
    tmp10 = tl.load(in_ptr1 + (r1 + (51328*x0)), rmask & tmp7 & xmask, other=0.0)
    tmp11 = tl.load(in_ptr2 + (r1 + (51328*x0)), rmask & tmp7 & xmask, other=0.0)
    tmp12 = tl.load(in_ptr3 + (tl.broadcast_to(r1, [XBLOCK, RBLOCK])), rmask & tmp7 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 + tmp12
    tmp14 = tmp10 + tmp13
    tmp15 = tl.full(tmp14.shape, 0.0, tmp14.dtype)
    tmp16 = tl.where(tmp7, tmp14, tmp15)
    tmp17 = tl.where(tmp3, tmp6, tmp16)
    tmp20 = tmp18 + tmp19
    tmp21 = tmp17 + tmp20
    tmp22 = tl.broadcast_to(tmp21, [XBLOCK, RBLOCK])
    tmp24 = tl.where(rmask & xmask, tmp22, 0)
    tmp25 = tl.broadcast_to(tmp22, [XBLOCK, RBLOCK])
    tmp27 = tl.where(rmask & xmask, tmp25, 0)
    tmp28 = tl.sum(tmp27, 1)[:, None]
    tmp29 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp30 = tmp29.to(tl.float32)
    tmp31 = tmp28 / tmp30
    tmp32 = tmp22 - tmp31
    tmp33 = tmp32 * tmp32
    tmp34 = tl.broadcast_to(tmp33, [XBLOCK, RBLOCK])
    tmp36 = tl.where(rmask & xmask, tmp34, 0)
    tmp37 = tl.sum(tmp36, 1)[:, None]
    tmp38 = tmp21 - tmp31
    tmp39 = 128.0
    tmp40 = tmp37 / tmp39
    tmp41 = 1e-06
    tmp42 = tmp40 + tmp41
    tmp43 = tl.math.rsqrt(tmp42)
    tmp44 = tmp38 * tmp43
    tmp46 = tmp44 * tmp45
    tmp48 = tmp46 + tmp47
    tmp49 = 0.5
    tmp50 = tmp48 * tmp49
    tmp51 = 0.7071067811865476
    tmp52 = tmp48 * tmp51
    tmp53 = tl.math.erf(tmp52)
    tmp54 = 1.0
    tmp55 = tmp53 + tmp54
    tmp56 = tmp50 * tmp55
    tl.store(in_out_ptr0 + (r1 + (128*x0)), tmp21, rmask & xmask)
    tl.store(in_out_ptr1 + (r1 + (128*x0)), tmp56, rmask & xmask)
