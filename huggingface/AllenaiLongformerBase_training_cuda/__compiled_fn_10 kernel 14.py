
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[256, 512], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_full_like_where_13', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 256
    xnumel = 257
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp9 = tl.load(in_ptr0 + (x1 + (513*y0)), xmask & ymask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr3 + (y0 + (1024*x1)), xmask & ymask, eviction_policy='evict_last')
    tmp0 = (-255) + x1 + y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 <= tmp1
    tmp3 = 1.0
    tmp4 = 0.0
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp6 = (tmp5 != 0)
    tmp7 = tl.full([1, 1], 0, tl.int32)
    tmp8 = tmp7 == tmp7
    tmp10 = tl.full([1, 1], 1, tl.int64)
    tmp11 = tmp1 >= tmp10
    tmp12 = tl.broadcast_to(x1, [XBLOCK, YBLOCK])
    tmp13 = tl.full([1, 1], 256, tl.int64)
    tmp14 = tmp12 < tmp13
    tmp15 = tmp14 & tmp11
    tmp16 = (((-131584) + x1 + (513*y0)) // 512) % 513
    tmp17 = tl.full([1, 1], 512, tl.int64)
    tmp18 = tmp16 < tmp17
    tmp19 = tmp18 & tmp15
    tmp20 = tl.load(in_ptr1 + ((256*((((-131584) + x1 + (513*y0)) // 262656) % 3)) + ((((-131584) + x1 + (513*y0)) // 512) % 513)), tmp19 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp21 = tl.load(in_ptr2 + ((256*((((-131584) + x1 + (513*y0)) // 262656) % 3)) + ((x1 + (513*y0)) % 512)), tmp19 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp22 = tmp20 * tmp21
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp19, tmp22, tmp23)
    tmp25 = tl.full(tmp24.shape, 0.0, tmp24.dtype)
    tmp26 = tl.where(tmp15, tmp24, tmp25)
    tmp27 = tl.load(in_ptr3 + (y0 + (1024*x1)), tmp11 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.where(tmp14, tmp26, tmp27)
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp11, tmp28, tmp29)
    tmp32 = tl.where(tmp11, tmp30, tmp31)
    tmp33 = tl.where(tmp8, tmp9, tmp32)
    tmp34 = float("-inf")
    tmp35 = tl.where(tmp6, tmp34, tmp33)
    tl.store(out_ptr0 + (x1 + (257*y0)), tmp35, xmask & ymask)
