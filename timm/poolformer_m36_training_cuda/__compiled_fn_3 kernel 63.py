
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_62', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2360
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r3 = rindex
    x0 = xindex % 59
    x1 = (xindex // 59) % 5
    x2 = (xindex // 295)
    x5 = xindex
    tmp0 = r3 + (128*x0)
    tmp1 = tl.full([1, 1], 7527, tl.int32)
    tmp2 = tmp0 < tmp1
    tmp3 = r3 + (128*x0) + (7527*x1)
    tmp4 = tl.full([1, 1], 37632, tl.int32)
    tmp5 = tmp3 < tmp4
    tmp6 = tmp5 & tmp2
    tmp7 = tl.load(in_ptr0 + ((768*((r3 + (128*x0) + (7527*x1)) % 49)) + (37632*x2) + (((r3 + (128*x0) + (7527*x1)) // 49) % 768)), rmask & tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp8 = tl.load(in_ptr1 + ((768*((r3 + (128*x0) + (7527*x1)) % 49)) + (37632*x2) + (((r3 + (128*x0) + (7527*x1)) // 49) % 768)), rmask & tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp9 = tl.load(in_ptr2 + ((768*((r3 + (128*x0) + (7527*x1)) % 49)) + (37632*x2) + (((r3 + (128*x0) + (7527*x1)) // 49) % 768)), rmask & tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp10 = tmp8 - tmp9
    tmp11 = tl.load(in_ptr3 + (((r3 + (128*x0) + (7527*x1)) // 49) % 768), rmask & tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp10 * tmp11
    tmp13 = tmp7 + tmp12
    tmp14 = tl.full(tmp13.shape, 0, tmp13.dtype)
    tmp15 = tl.where(tmp6, tmp13, tmp14)
    tmp16 = tl.full(tmp15.shape, 0, tmp15.dtype)
    tmp17 = tl.where(tmp2, tmp15, tmp16)
    tmp18 = 0.0
    tmp19 = tl.full(tmp18.shape, 0, tmp18.dtype)
    tmp20 = tl.where(tmp6, tmp18, tmp19)
    tmp21 = tl.full(tmp20.shape, 0, tmp20.dtype)
    tmp22 = tl.where(tmp2, tmp20, tmp21)
    tmp23 = 1.0
    tmp24 = tl.full(tmp23.shape, 0, tmp23.dtype)
    tmp25 = tl.where(tmp6, tmp23, tmp24)
    tmp26 = tl.full(tmp25.shape, 0, tmp25.dtype)
    tmp27 = tl.where(tmp2, tmp25, tmp26)
    tmp28 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
    tmp29 = tl.broadcast_to(tmp22, [XBLOCK, RBLOCK])
    tmp30 = tl.broadcast_to(tmp27, [XBLOCK, RBLOCK])
    tmp32 = tl.where(rmask & xmask, tmp28, 0)
    tmp33 = tl.where(rmask & xmask, tmp29, 0)
    tmp34 = tl.where(rmask & xmask, tmp30, 0)
    tmp35, tmp36, tmp37 = triton_helpers.welford(tmp32, tmp33, tmp34, 1)
    tmp38 = tmp35[:, None]
    tmp39 = tmp36[:, None]
    tmp40 = tmp37[:, None]
    tl.store(out_ptr0 + (x5), tmp38, xmask)
    tl.store(out_ptr1 + (x5), tmp39, xmask)
    tl.store(out_ptr2 + (x5), tmp40, xmask)
