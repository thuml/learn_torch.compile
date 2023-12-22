
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32768, 64],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_cat_convolution_mean_relu_56', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 21504
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    x0 = xindex % 2688
    r2 = rindex
    x1 = (xindex // 2688)
    x3 = xindex
    tmp58 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp60 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp68 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp70 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    tmp0 = x0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 2048, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (r2 + (49*x0) + (112896*x1)), rmask & tmp4 & xmask, other=0.0)
    tmp6 = tl.load(in_ptr1 + (r2 + (49*x0) + (106624*x1)), rmask & tmp4 & xmask, other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.load(in_ptr2 + (r2 + (49*x0) + (106624*x1)), rmask & tmp4 & xmask, other=0.0)
    tmp9 = tmp7 + tmp8
    tmp10 = tl.load(in_ptr3 + (r2 + (49*x0) + (106624*x1)), rmask & tmp4 & xmask, other=0.0)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp4, tmp11, tmp12)
    tmp14 = tmp0 >= tmp3
    tmp15 = tl.full([1, 1], 2688, tl.int64)
    tmp16 = tmp0 < tmp15
    tmp17 = tl.broadcast_to((-2048) + x0, [XBLOCK, RBLOCK])
    tmp18 = tmp17 >= tmp1
    tmp19 = tl.full([1, 1], 512, tl.int64)
    tmp20 = tmp17 < tmp19
    tmp21 = tmp20 & tmp14
    tmp22 = tl.full([1, 1], 384, tl.int64)
    tmp23 = tmp17 < tmp22
    tmp24 = tmp23 & tmp21
    tmp25 = tl.full([1, 1], 256, tl.int64)
    tmp26 = tmp17 < tmp25
    tmp27 = tmp26 & tmp24
    tmp28 = tl.load(in_ptr0 + (r2 + (49*x0) + (112896*x1)), rmask & tmp27 & xmask, other=0.0)
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp27, tmp28, tmp29)
    tmp31 = tmp17 >= tmp25
    tmp32 = tmp31 & tmp24
    tmp33 = tl.load(in_ptr1 + ((-12544) + r2 + (49*x0) + (106624*x1)), rmask & tmp32 & xmask, other=0.0)
    tmp34 = tl.full(tmp33.shape, 0.0, tmp33.dtype)
    tmp35 = tl.where(tmp32, tmp33, tmp34)
    tmp36 = tl.where(tmp26, tmp30, tmp35)
    tmp37 = tl.full(tmp36.shape, 0.0, tmp36.dtype)
    tmp38 = tl.where(tmp24, tmp36, tmp37)
    tmp39 = tmp17 >= tmp22
    tmp40 = tmp39 & tmp21
    tmp41 = tl.load(in_ptr2 + ((-18816) + r2 + (49*x0) + (106624*x1)), rmask & tmp40 & xmask, other=0.0)
    tmp42 = tl.full(tmp41.shape, 0.0, tmp41.dtype)
    tmp43 = tl.where(tmp40, tmp41, tmp42)
    tmp44 = tl.where(tmp23, tmp38, tmp43)
    tmp45 = tl.full(tmp44.shape, 0.0, tmp44.dtype)
    tmp46 = tl.where(tmp21, tmp44, tmp45)
    tmp47 = tmp17 >= tmp19
    tmp48 = tl.full([1, 1], 640, tl.int64)
    tmp49 = tmp17 < tmp48
    tmp50 = tmp47 & tmp14
    tmp51 = tl.load(in_ptr3 + ((-25088) + r2 + (49*x0) + (106624*x1)), rmask & tmp50 & xmask, other=0.0)
    tmp52 = tl.full(tmp51.shape, 0.0, tmp51.dtype)
    tmp53 = tl.where(tmp50, tmp51, tmp52)
    tmp54 = tl.where(tmp20, tmp46, tmp53)
    tmp55 = tl.full(tmp54.shape, 0.0, tmp54.dtype)
    tmp56 = tl.where(tmp14, tmp54, tmp55)
    tmp57 = tl.where(tmp4, tmp13, tmp56)
    tmp59 = tmp57 - tmp58
    tmp61 = 0.001
    tmp62 = tmp60 + tmp61
    tmp63 = tl.sqrt(tmp62)
    tmp64 = 1 / tmp63
    tmp65 = 1.0
    tmp66 = tmp64 * tmp65
    tmp67 = tmp59 * tmp66
    tmp69 = tmp67 * tmp68
    tmp71 = tmp69 + tmp70
    tmp72 = triton_helpers.maximum(0, tmp71)
    tmp73 = tl.broadcast_to(tmp72, [XBLOCK, RBLOCK])
    tmp75 = tl.where(rmask & xmask, tmp73, 0)
    tmp76 = tl.sum(tmp75, 1)[:, None]
    tmp77 = 49.0
    tmp78 = tmp76 / tmp77
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp78, xmask)
