
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32768, 64],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_backward_174', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    x0 = xindex % 3136
    r2 = rindex
    x1 = (xindex // 3136)
    x3 = xindex
    tmp16 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp22 = tl.load(in_ptr3 + (r2 + (64*x3)), rmask & xmask, other=0.0)
    tmp28 = tl.load(in_ptr4 + (x3), xmask, eviction_policy='evict_last')
    tmp0 = 1 + x0
    tmp1 = tl.full([1, 1], 1, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + (64 + r2 + (64*x0) + (200768*x1)), rmask & tmp2 & xmask, other=0.0)
    tmp4 = tl.load(in_ptr1 + (x0 + (3136*r2) + (200704*x1)), rmask & tmp2 & xmask, other=0.0)
    tmp5 = tmp3 + tmp4
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp2, tmp5, tmp6)
    tmp8 = 0.0
    tmp9 = tl.where(tmp2, tmp7, tmp8)
    tmp10 = tmp0 < tmp1
    tmp11 = tl.load(in_ptr0 + (r2 + (200768*x1)), rmask & tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp14 = tl.where(tmp10, tmp13, tmp8)
    tmp15 = tmp9 + tmp14
    tmp17 = tmp15 * tmp16
    tmp18 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
    tmp20 = tl.where(rmask & xmask, tmp18, 0)
    tmp21 = tl.sum(tmp20, 1)[:, None]
    tmp23 = tmp17 * tmp22
    tmp24 = tl.broadcast_to(tmp23, [XBLOCK, RBLOCK])
    tmp26 = tl.where(rmask & xmask, tmp24, 0)
    tmp27 = tl.sum(tmp26, 1)[:, None]
    tmp29 = 64.0
    tmp30 = tmp17 * tmp29
    tmp31 = tmp30 - tmp21
    tmp32 = tmp22 * tmp27
    tmp33 = tmp31 - tmp32
    tmp34 = tmp28 * tmp33
    tl.store(out_ptr2 + (r2 + (64*x3)), tmp34, rmask & xmask)
