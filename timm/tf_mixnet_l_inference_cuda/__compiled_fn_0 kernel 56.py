
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_cat_mean_silu_55', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 4992
    rnumel = 196
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    x0 = xindex % 624
    r2 = rindex
    x1 = (xindex // 624)
    x3 = xindex
    tmp31 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp43 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    tmp0 = x0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 156, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (r2 + (196*x0) + (30576*x1)), rmask & tmp4 & xmask, other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 312, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tmp8 & tmp10
    tmp12 = tl.load(in_ptr1 + ((-30576) + r2 + (196*x0) + (30576*x1)), rmask & tmp11 & xmask, other=0.0)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp11, tmp12, tmp13)
    tmp15 = tmp0 >= tmp9
    tmp16 = tl.full([1, 1], 468, tl.int64)
    tmp17 = tmp0 < tmp16
    tmp18 = tmp15 & tmp17
    tmp19 = tl.load(in_ptr2 + ((-61152) + r2 + (196*x0) + (30576*x1)), rmask & tmp18 & xmask, other=0.0)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = tmp0 >= tmp16
    tmp23 = tl.full([1, 1], 624, tl.int64)
    tmp24 = tmp0 < tmp23
    tmp25 = tl.load(in_ptr3 + ((-91728) + r2 + (196*x0) + (30576*x1)), rmask & tmp22 & xmask, other=0.0)
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp22, tmp25, tmp26)
    tmp28 = tl.where(tmp18, tmp21, tmp27)
    tmp29 = tl.where(tmp11, tmp14, tmp28)
    tmp30 = tl.where(tmp4, tmp7, tmp29)
    tmp32 = tmp30 - tmp31
    tmp34 = 0.001
    tmp35 = tmp33 + tmp34
    tmp36 = tl.sqrt(tmp35)
    tmp37 = 1 / tmp36
    tmp38 = 1.0
    tmp39 = tmp37 * tmp38
    tmp40 = tmp32 * tmp39
    tmp42 = tmp40 * tmp41
    tmp44 = tmp42 + tmp43
    tmp45 = tl.sigmoid(tmp44)
    tmp46 = tmp44 * tmp45
    tmp47 = tl.broadcast_to(tmp46, [XBLOCK, RBLOCK])
    tmp49 = tl.where(rmask & xmask, tmp47, 0)
    tmp50 = tl.sum(tmp49, 1)[:, None]
    tmp51 = 196.0
    tmp52 = tmp50 / tmp51
    tl.store(out_ptr0 + (r2 + (196*x3)), tmp44, rmask & xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp52, xmask)
