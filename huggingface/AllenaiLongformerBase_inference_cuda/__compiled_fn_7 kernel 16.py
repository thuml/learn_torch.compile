
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[16384, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_add_15', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 12288
    XBLOCK: tl.constexpr = 1
    rnumel = 513
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    x1 = (xindex // 12)
    r2 = rindex
    x0 = xindex % 12
    x3 = xindex
    tmp23 = tl.load(in_ptr0 + (r2 + (513*x0) + (6156*(x1 % 256)) + (1575936*((((256*(x1 // 256)) + (x1 % 256)) // 256) % 4))), rmask, other=0.0)
    tmp33 = tl.load(in_ptr1 + (r2 + (513*x1)), rmask, eviction_policy='evict_last', other=0.0)
    tmp0 = x1
    tmp1 = tl.full([1], 768, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.broadcast_to(r2, [RBLOCK])
    tmp4 = tl.full([1], 256, tl.int64)
    tmp5 = tmp3 >= tmp4
    tmp6 = tmp5 & tmp2
    tmp7 = 1280 + ((-1)*r2) + ((-1)*x1)
    tmp8 = tl.full([1], 0, tl.int64)
    tmp9 = tmp7 <= tmp8
    tmp10 = 1.0
    tmp11 = 0.0
    tmp12 = tl.where(tmp9, tmp10, tmp11)
    tmp13 = (tmp12 != 0)
    tmp14 = tl.load(in_ptr0 + (r2 + (513*x0) + (6156*(x1 % 256)) + (1575936*((((256*(x1 // 256)) + (x1 % 256)) // 256) % 4))), rmask & tmp6, other=0.0)
    tmp15 = float("-inf")
    tmp16 = tl.where(tmp13, tmp15, tmp14)
    tmp17 = tl.full(tmp16.shape, 0.0, tmp16.dtype)
    tmp18 = tl.where(tmp6, tmp16, tmp17)
    tmp19 = tl.load(in_ptr0 + (r2 + (513*x0) + (6156*(x1 % 256)) + (1575936*((((256*(x1 // 256)) + (x1 % 256)) // 256) % 4))), rmask & tmp2, other=0.0)
    tmp20 = tl.where(tmp5, tmp18, tmp19)
    tmp21 = tl.full(tmp20.shape, 0.0, tmp20.dtype)
    tmp22 = tl.where(tmp2, tmp20, tmp21)
    tmp24 = tl.where(tmp2, tmp22, tmp23)
    tmp25 = tl.load(in_ptr1 + (r2 + (513*x1)), rmask & tmp6, eviction_policy='evict_last', other=0.0)
    tmp26 = tl.where(tmp13, tmp15, tmp25)
    tmp27 = tl.full(tmp26.shape, 0.0, tmp26.dtype)
    tmp28 = tl.where(tmp6, tmp26, tmp27)
    tmp29 = tl.load(in_ptr1 + (r2 + (513*x1)), rmask & tmp2, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.where(tmp5, tmp28, tmp29)
    tmp31 = tl.full(tmp30.shape, 0.0, tmp30.dtype)
    tmp32 = tl.where(tmp2, tmp30, tmp31)
    tmp34 = tl.where(tmp2, tmp32, tmp33)
    tmp35 = tmp24 + tmp34
    tmp36 = tl.broadcast_to(tmp35, [RBLOCK])
    tmp38 = tl.where(rmask, tmp36, float("-inf"))
    tmp39 = triton_helpers.promote_to_tensor(triton_helpers.max2(tmp38, 0))
    tmp40 = tmp35 - tmp39
    tmp41 = tl.exp(tmp40)
    tmp42 = tl.broadcast_to(tmp41, [RBLOCK])
    tmp44 = tl.where(rmask, tmp42, 0)
    tmp45 = triton_helpers.promote_to_tensor(tl.sum(tmp44, 0))
    tl.store(out_ptr0 + (r2 + (513*x3)), tmp35, rmask)
    tl.store(out_ptr1 + (x3), tmp39, None)
    tl.store(out_ptr2 + (x3), tmp45, None)
