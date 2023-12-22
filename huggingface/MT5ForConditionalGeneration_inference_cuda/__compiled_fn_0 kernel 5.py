
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_4', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 768
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 128
    x1 = (xindex // 128)
    tmp0 = tl.load(in_ptr0 + (r2 + (128*x3)), rmask & xmask, other=0.0)
    tmp1 = r2 + ((-1)*x0)
    tmp2 = tl.full([1, 1], 0, tl.int64)
    tmp3 = tmp1 > tmp2
    tmp4 = tmp3.to(tl.int64)
    tmp5 = tl.full([1, 1], 16, tl.int64)
    tmp6 = tmp4 * tmp5
    tmp7 = tmp6 + tmp2
    tmp8 = tl.abs(tmp1)
    tmp9 = tl.full([1, 1], 8, tl.int64)
    tmp10 = tmp8 < tmp9
    tmp11 = tmp8.to(tl.float32)
    tmp12 = 8.0
    tmp13 = tmp11 / tmp12
    tmp14 = tl.log(tmp13)
    tmp15 = 2.772588722239781
    tmp16 = tmp14 / tmp15
    tmp17 = tmp16 * tmp12
    tmp18 = tmp17.to(tl.int64)
    tmp19 = tmp18 + tmp9
    tmp20 = tl.full([1, 1], 15, tl.int64)
    tmp21 = triton_helpers.minimum(tmp19, tmp20)
    tmp22 = tl.where(tmp10, tmp8, tmp21)
    tmp23 = tmp7 + tmp22
    tmp24 = tmp23 + 32
    tmp25 = tmp23 < 0
    tmp26 = tl.where(tmp25, tmp24, tmp23)
    tl.device_assert(((0 <= tmp26) & (tmp26 < 32)) | ~(rmask & xmask), "index out of bounds: 0 <= tmp26 < 32")
    tmp27 = tl.load(in_ptr1 + (x1 + (6*tmp26)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tmp0 + tmp27
    tmp29 = tl.broadcast_to(tmp28, [XBLOCK, RBLOCK])
    tmp31 = tl.where(rmask & xmask, tmp29, float("-inf"))
    tmp32 = triton_helpers.max2(tmp31, 1)[:, None]
    tmp33 = tmp28 - tmp32
    tmp34 = tl.exp(tmp33)
    tmp35 = tl.broadcast_to(tmp34, [XBLOCK, RBLOCK])
    tmp37 = tl.where(rmask & xmask, tmp35, 0)
    tmp38 = tl.sum(tmp37, 1)[:, None]
    tmp39 = tmp34 / tmp38
    tl.store(out_ptr2 + (r2 + (128*x3)), tmp39, rmask & xmask)
