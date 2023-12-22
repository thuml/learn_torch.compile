
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_40', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 4720
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r3 = rindex
    x1 = (xindex // 10) % 59
    x0 = xindex % 10
    x2 = (xindex // 590)
    x4 = xindex
    tmp0 = r3 + (128*x1)
    tmp1 = tl.full([1, 1], 7527, tl.int32)
    tmp2 = tmp0 < tmp1
    tmp3 = r3 + (128*x1) + (7527*x0)
    tmp4 = tl.full([1, 1], 75264, tl.int32)
    tmp5 = tmp3 < tmp4
    tmp6 = tmp5 & tmp2
    tmp7 = tl.load(in_ptr0 + ((384*((r3 + (128*x1) + (7527*x0)) % 196)) + (75264*x2) + (((r3 + (128*x1) + (7527*x0)) // 196) % 384)), rmask & tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp8 = tl.full(tmp7.shape, 0, tmp7.dtype)
    tmp9 = tl.where(tmp6, tmp7, tmp8)
    tmp10 = tl.full(tmp9.shape, 0, tmp9.dtype)
    tmp11 = tl.where(tmp2, tmp9, tmp10)
    tmp12 = 0.0
    tmp13 = tl.full(tmp12.shape, 0, tmp12.dtype)
    tmp14 = tl.where(tmp6, tmp12, tmp13)
    tmp15 = tl.full(tmp14.shape, 0, tmp14.dtype)
    tmp16 = tl.where(tmp2, tmp14, tmp15)
    tmp17 = 1.0
    tmp18 = tl.full(tmp17.shape, 0, tmp17.dtype)
    tmp19 = tl.where(tmp6, tmp17, tmp18)
    tmp20 = tl.full(tmp19.shape, 0, tmp19.dtype)
    tmp21 = tl.where(tmp2, tmp19, tmp20)
    tmp22 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
    tmp23 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
    tmp24 = tl.broadcast_to(tmp21, [XBLOCK, RBLOCK])
    tmp26 = tl.where(rmask & xmask, tmp22, 0)
    tmp27 = tl.where(rmask & xmask, tmp23, 0)
    tmp28 = tl.where(rmask & xmask, tmp24, 0)
    tmp29, tmp30, tmp31 = triton_helpers.welford(tmp26, tmp27, tmp28, 1)
    tmp32 = tmp29[:, None]
    tmp33 = tmp30[:, None]
    tmp34 = tmp31[:, None]
    tl.store(out_ptr0 + (x4), tmp32, xmask)
    tl.store(out_ptr1 + (x4), tmp33, xmask)
    tl.store(out_ptr2 + (x4), tmp34, xmask)
