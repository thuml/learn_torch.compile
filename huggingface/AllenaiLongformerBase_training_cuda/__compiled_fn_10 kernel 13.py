
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[256, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_copy_slice_scatter_12', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 256
    xnumel = 513
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex
    x1 = xindex
    tmp56 = tl.load(in_ptr2 + (y0 + (1024*x1)), xmask & ymask, eviction_policy='evict_last')
    tmp0 = y0
    tmp1 = tl.full([1, 1], 1, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.broadcast_to(x1, [XBLOCK, YBLOCK])
    tmp4 = tmp3 >= tmp1
    tmp5 = tl.full([1, 1], 256, tl.int64)
    tmp6 = tmp3 < tmp5
    tmp7 = tmp4 & tmp6
    tmp8 = tmp7 & tmp2
    tmp9 = (((-256) + x1 + (513*y0)) // 512) % 513
    tmp10 = tl.full([1, 1], 512, tl.int64)
    tmp11 = tmp9 < tmp10
    tmp12 = tmp11 & tmp8
    tmp13 = tl.load(in_ptr0 + ((256*((((-256) + x1 + (513*y0)) // 262656) % 3)) + ((((-256) + x1 + (513*y0)) // 512) % 513)), tmp12 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp14 = tl.load(in_ptr1 + ((256*((((-256) + x1 + (513*y0)) // 262656) % 3)) + (((-256) + x1 + (513*y0)) % 512)), tmp12 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp15 = tmp13 * tmp14
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp12, tmp15, tmp16)
    tmp18 = tl.full(tmp17.shape, 0.0, tmp17.dtype)
    tmp19 = tl.where(tmp8, tmp17, tmp18)
    tmp20 = tl.full([1, 1], 0, tl.int64)
    tmp21 = tmp20 >= tmp1
    tmp22 = tmp21 & tmp2
    tmp23 = tmp6 & tmp22
    tmp24 = (((-131584) + x1 + (513*y0)) // 512) % 513
    tmp25 = tmp24 < tmp10
    tmp26 = tmp25 & tmp23
    tmp27 = tl.load(in_ptr0 + ((256*((((-131584) + x1 + (513*y0)) // 262656) % 3)) + ((((-131584) + x1 + (513*y0)) // 512) % 513)), tmp26 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.load(in_ptr1 + ((256*((((-131584) + x1 + (513*y0)) // 262656) % 3)) + ((x1 + (513*y0)) % 512)), tmp26 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp29 = tmp27 * tmp28
    tmp30 = tl.full(tmp29.shape, 0.0, tmp29.dtype)
    tmp31 = tl.where(tmp26, tmp29, tmp30)
    tmp32 = tl.full(tmp31.shape, 0.0, tmp31.dtype)
    tmp33 = tl.where(tmp23, tmp31, tmp32)
    tmp34 = tl.load(in_ptr2 + (y0 + (1024*x1)), tmp22 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp35 = tl.where(tmp6, tmp33, tmp34)
    tmp36 = tl.full(tmp35.shape, 0.0, tmp35.dtype)
    tmp37 = tl.where(tmp22, tmp35, tmp36)
    tmp38 = tl.load(in_ptr2 + (y0 + (1024*x1)), tmp2 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp39 = tl.where(tmp21, tmp37, tmp38)
    tmp40 = tl.where(tmp7, tmp19, tmp39)
    tmp41 = tl.full(tmp40.shape, 0.0, tmp40.dtype)
    tmp42 = tl.where(tmp2, tmp40, tmp41)
    tmp43 = tmp6 & tmp21
    tmp44 = tmp25 & tmp43
    tmp45 = tl.load(in_ptr0 + ((256*((((-131584) + x1 + (513*y0)) // 262656) % 3)) + ((((-131584) + x1 + (513*y0)) // 512) % 513)), tmp44 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp46 = tl.load(in_ptr1 + ((256*((((-131584) + x1 + (513*y0)) // 262656) % 3)) + ((x1 + (513*y0)) % 512)), tmp44 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp47 = tmp45 * tmp46
    tmp48 = tl.full(tmp47.shape, 0.0, tmp47.dtype)
    tmp49 = tl.where(tmp44, tmp47, tmp48)
    tmp50 = tl.full(tmp49.shape, 0.0, tmp49.dtype)
    tmp51 = tl.where(tmp43, tmp49, tmp50)
    tmp52 = tl.load(in_ptr2 + (y0 + (1024*x1)), tmp21 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp53 = tl.where(tmp6, tmp51, tmp52)
    tmp54 = tl.full(tmp53.shape, 0.0, tmp53.dtype)
    tmp55 = tl.where(tmp21, tmp53, tmp54)
    tmp57 = tl.where(tmp21, tmp55, tmp56)
    tmp58 = tl.where(tmp2, tmp42, tmp57)
    tl.store(out_ptr0 + (x1 + (513*y0)), tmp58, xmask & ymask)
