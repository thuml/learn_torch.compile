
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*i64', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_clone_copy_fill_lift_fresh_new_zeros_nll_loss_forward_select_scatter_slice_scatter_5', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r0 = rindex
    tmp0 = r0
    tmp1 = tl.full([1, 1], 127, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp3 = tl.full([1, 1], 127, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (tl.broadcast_to(1 + r0, [XBLOCK, RBLOCK])), rmask & tmp4, other=0.0)
    tmp6 = tl.full(tmp5.shape, 0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tl.full([1, 1], 0, tl.int64)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tl.full([1, 1], 1, tl.int64)
    tmp11 = tl.where(tmp2, tmp10, tmp9)
    tmp12 = tl.full([1, 1], -100, tl.int64)
    tmp13 = tmp11 != tmp12
    tmp14 = tmp13.to(tl.int64)
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
    tmp17 = tl.where(rmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None]
    tmp19 = tl.where(tmp13, tmp11, tmp8)
    tmp20 = tmp19 + 256008
    tmp21 = tmp19 < 0
    tmp22 = tl.where(tmp21, tmp20, tmp19)
    tl.device_assert((0 <= tmp22) & (tmp22 < 256008), "index out of bounds: 0 <= tmp22 < 256008")
    tmp23 = tl.load(in_ptr1 + (tmp22 + (256008*r0)), rmask, eviction_policy='evict_last', other=0.0)
    tmp24 = -tmp23
    tmp25 = 0.0
    tmp26 = tl.where(tmp13, tmp24, tmp25)
    tmp27 = tl.broadcast_to(tmp26, [XBLOCK, RBLOCK])
    tmp29 = tl.where(rmask, tmp27, 0)
    tmp30 = tl.sum(tmp29, 1)[:, None]
    tmp31 = tmp18.to(tl.float32)
    tmp32 = tmp30 / tmp31
    tl.store(out_ptr0 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp11, rmask)
    tl.store(out_ptr2 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp31, None)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp32, None)
