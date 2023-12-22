
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_copy_slice_scatter_14', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 513
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex
    x1 = xindex
    tmp42 = tl.load(in_ptr1 + (x1 + (513*(y0 % 256))), xmask, eviction_policy='evict_last')
    tmp57 = tl.load(in_ptr4 + (y0 + (1024*x1)), xmask, eviction_policy='evict_last')
    tmp0 = y0
    tmp1 = tl.full([1, 1], 256, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.broadcast_to(x1, [XBLOCK, YBLOCK])
    tmp4 = tl.full([1, 1], 257, tl.int64)
    tmp5 = tmp3 < tmp4
    tmp6 = tmp5 & tmp2
    tmp7 = tl.load(in_ptr0 + (x1 + (257*y0)), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp6, tmp7, tmp8)
    tmp10 = tl.broadcast_to((y0 // 256), [XBLOCK, YBLOCK])
    tmp11 = tl.full([1, 1], 0, tl.int32)
    tmp12 = tmp10 == tmp11
    tmp13 = tl.load(in_ptr1 + (x1 + (513*(y0 % 256))), tmp2 & xmask, eviction_policy='evict_last', other=0.0)
    tmp14 = tl.full([1, 1], 1, tl.int64)
    tmp15 = tmp10 >= tmp14
    tmp16 = tmp15 & tmp2
    tmp17 = tmp3 < tmp1
    tmp18 = tmp17 & tmp16
    tmp19 = (((-131584) + x1 + (513*(y0 % 256)) + (262656*(y0 // 256))) // 512) % 513
    tmp20 = tl.full([1, 1], 512, tl.int64)
    tmp21 = tmp19 < tmp20
    tmp22 = tmp21 & tmp18
    tmp23 = tl.load(in_ptr2 + ((256*((((-131584) + x1 + (513*(y0 % 256)) + (262656*(y0 // 256))) // 262656) % 3)) + ((((-131584) + x1 + (513*(y0 % 256)) + (262656*(y0 // 256))) // 512) % 513)), tmp22 & xmask, eviction_policy='evict_last', other=0.0)
    tmp24 = tl.load(in_ptr3 + ((256*((((-131584) + x1 + (513*(y0 % 256)) + (262656*(y0 // 256))) // 262656) % 3)) + ((x1 + (513*(y0 % 256))) % 512)), tmp22 & xmask, eviction_policy='evict_last', other=0.0)
    tmp25 = tmp23 * tmp24
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp22, tmp25, tmp26)
    tmp28 = tl.full(tmp27.shape, 0.0, tmp27.dtype)
    tmp29 = tl.where(tmp18, tmp27, tmp28)
    tmp30 = tl.load(in_ptr4 + (y0 + (1024*x1)), tmp16 & xmask, eviction_policy='evict_last', other=0.0)
    tmp31 = tl.where(tmp17, tmp29, tmp30)
    tmp32 = tl.full(tmp31.shape, 0.0, tmp31.dtype)
    tmp33 = tl.where(tmp16, tmp31, tmp32)
    tmp34 = tl.load(in_ptr4 + (y0 + (1024*x1)), tmp2 & xmask, eviction_policy='evict_last', other=0.0)
    tmp35 = tl.where(tmp15, tmp33, tmp34)
    tmp36 = tl.where(tmp12, tmp13, tmp35)
    tmp37 = tl.where(tmp5, tmp9, tmp36)
    tmp38 = tl.full(tmp37.shape, 0.0, tmp37.dtype)
    tmp39 = tl.where(tmp2, tmp37, tmp38)
    tmp40 = (y0 // 256)
    tmp41 = tmp40 == tmp11
    tmp43 = tmp40 >= tmp14
    tmp44 = tmp17 & tmp43
    tmp45 = tmp21 & tmp44
    tmp46 = tl.load(in_ptr2 + ((256*((((-131584) + x1 + (513*(y0 % 256)) + (262656*(y0 // 256))) // 262656) % 3)) + ((((-131584) + x1 + (513*(y0 % 256)) + (262656*(y0 // 256))) // 512) % 513)), tmp45 & xmask, eviction_policy='evict_last', other=0.0)
    tmp47 = tl.load(in_ptr3 + ((256*((((-131584) + x1 + (513*(y0 % 256)) + (262656*(y0 // 256))) // 262656) % 3)) + ((x1 + (513*(y0 % 256))) % 512)), tmp45 & xmask, eviction_policy='evict_last', other=0.0)
    tmp48 = tmp46 * tmp47
    tmp49 = tl.full(tmp48.shape, 0.0, tmp48.dtype)
    tmp50 = tl.where(tmp45, tmp48, tmp49)
    tmp51 = tl.full(tmp50.shape, 0.0, tmp50.dtype)
    tmp52 = tl.where(tmp44, tmp50, tmp51)
    tmp53 = tl.load(in_ptr4 + (y0 + (1024*x1)), tmp43 & xmask, eviction_policy='evict_last', other=0.0)
    tmp54 = tl.where(tmp17, tmp52, tmp53)
    tmp55 = tl.full(tmp54.shape, 0.0, tmp54.dtype)
    tmp56 = tl.where(tmp43, tmp54, tmp55)
    tmp58 = tl.where(tmp43, tmp56, tmp57)
    tmp59 = tl.where(tmp41, tmp42, tmp58)
    tmp60 = tl.where(tmp2, tmp39, tmp59)
    tl.store(out_ptr0 + (x1 + (513*y0)), tmp60, xmask)
