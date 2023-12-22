
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[4096, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_embedding_mean_mul_pow_rsqrt_view_0', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel):
    xnumel = 4096
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    x0 = xindex
    r1 = rindex
    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tmp0 + 32128
    tmp2 = tmp0 < 0
    tmp3 = tl.where(tmp2, tmp1, tmp0)
    tl.device_assert((0 <= tmp3) & (tmp3 < 32128), "index out of bounds: 0 <= tmp3 < 32128")
    tmp4 = tl.load(in_ptr1 + (r1 + (512*tmp3)), rmask, other=0.0)
    tmp5 = tmp4 * tmp4
    tmp6 = tl.broadcast_to(tmp5, [RBLOCK])
    tmp8 = tl.where(rmask, tmp6, 0)
    tmp9 = triton_helpers.promote_to_tensor(tl.sum(tmp8, 0))
    tmp10 = 512.0
    tmp11 = tmp9 / tmp10
    tmp12 = 1e-06
    tmp13 = tmp11 + tmp12
    tmp14 = tl.math.rsqrt(tmp13)
    tmp16 = tmp4 * tmp14
    tmp17 = tmp15 * tmp16
    tl.store(out_ptr0 + (r1 + (512*x0)), tmp4, rmask)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp14, None)
    tl.store(out_ptr1 + (r1 + (512*x0)), tmp17, rmask)
