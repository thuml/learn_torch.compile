
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[4096, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_add_mul_53', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = 144
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 16
    x1 = (xindex // 16)
    tmp0 = tl.load(in_ptr0 + (r2 + (144*x3)), rmask, other=0.0)
    tmp1 = 0.25
    tmp2 = tmp0 * tmp1
    tmp3 = 11 + (23*(x0 // 4)) + (r2 // 12)
    tmp4 = tl.full([1, 1], 96, tl.int64)
    tmp5 = tmp3 < tmp4
    tmp6 = (11 + (23*(x0 // 4)) + (r2 // 12)) % 24
    tmp7 = tl.full([1, 1], 23, tl.int64)
    tmp8 = tmp6 < tmp7
    tmp9 = tmp8 & tmp5
    tmp10 = tl.load(in_ptr1 + ((23*((11 + (23*(x0 // 4)) + (r2 // 12)) // 24)) + (92*(x0 % 4)) + (368*x1) + ((11 + (23*(x0 // 4)) + (r2 // 12)) % 24)), rmask & tmp9, eviction_policy='evict_last', other=0.0)
    tmp11 = tl.full(tmp10.shape, 0.0, tmp10.dtype)
    tmp12 = tl.where(tmp9, tmp10, tmp11)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp5, tmp12, tmp13)
    tmp15 = 11 + (23*(x0 % 4)) + (r2 % 12)
    tmp16 = tmp15 < tmp4
    tmp17 = (11 + (23*(x0 % 4)) + (r2 % 12)) % 24
    tmp18 = tmp17 < tmp7
    tmp19 = tmp18 & tmp16
    tmp20 = tl.load(in_ptr2 + ((23*(((11 + (23*(x0 % 4)) + (r2 % 12)) // 24) % 4)) + (92*(x0 // 4)) + (368*x1) + ((11 + (23*(x0 % 4)) + (r2 % 12)) % 24)), rmask & tmp19, eviction_policy='evict_last', other=0.0)
    tmp21 = tl.full(tmp20.shape, 0.0, tmp20.dtype)
    tmp22 = tl.where(tmp19, tmp20, tmp21)
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp16, tmp22, tmp23)
    tmp25 = tmp14 + tmp24
    tmp26 = tmp2 + tmp25
    tmp27 = tl.broadcast_to(tmp26, [XBLOCK, RBLOCK])
    tmp29 = tl.where(rmask, tmp27, float("-inf"))
    tmp30 = triton_helpers.max2(tmp29, 1)[:, None]
    tmp31 = tmp26 - tmp30
    tmp32 = tl.exp(tmp31)
    tmp33 = tl.broadcast_to(tmp32, [XBLOCK, RBLOCK])
    tmp35 = tl.where(rmask, tmp33, 0)
    tmp36 = tl.sum(tmp35, 1)[:, None]
    tmp37 = tmp32 / tmp36
    tl.store(out_ptr2 + (r2 + (144*x3)), tmp37, rmask)
