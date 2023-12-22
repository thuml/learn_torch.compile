
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_copy_new_zeros_select_scatter_slice_scatter_11', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 513
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = (xindex // 256)
    y0 = yindex
    x1 = xindex % 256
    x3 = xindex
    tmp0 = x2
    tmp1 = tl.full([1, 1], 3, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp3 = y0
    tmp4 = tl.full([1, 1], 256, tl.int64)
    tmp5 = tmp3 >= tmp4
    tmp6 = ((656384 + y0 + (513*x1)) // 512) % 513
    tmp7 = tl.full([1, 1], 512, tl.int64)
    tmp8 = tmp6 < tmp7
    tmp9 = tmp8 & tmp5
    tmp10 = tl.load(in_ptr0 + ((256*((656384 + y0 + (513*x1)) // 262656)) + (((656384 + y0 + (513*x1)) // 512) % 513)), tmp9 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp11 = tl.load(in_ptr1 + ((256*((656384 + y0 + (513*x1)) // 262656)) + ((y0 + (513*x1)) % 512)), tmp9 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp10 * tmp11
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp9, tmp12, tmp13)
    tmp15 = tl.full(tmp14.shape, 0.0, tmp14.dtype)
    tmp16 = tl.where(tmp5, tmp14, tmp15)
    tmp17 = tl.full([1, 1], 3, tl.int64)
    tmp18 = tmp17 < tmp17
    tmp19 = tl.broadcast_to(y0, [XBLOCK, YBLOCK])
    tmp20 = tmp19 >= tmp4
    tmp21 = tmp20 & tmp18
    tmp22 = ((787712 + y0 + (513*x1)) // 512) % 513
    tmp23 = tmp22 < tmp7
    tmp24 = tmp23 & tmp21
    tmp25 = tl.load(in_ptr0 + ((256*(((787712 + y0 + (513*x1)) // 262656) % 3)) + (((787712 + y0 + (513*x1)) // 512) % 513)), tmp24 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp26 = tl.load(in_ptr1 + ((256*(((787712 + y0 + (513*x1)) // 262656) % 3)) + ((787712 + y0 + (513*x1)) % 512)), tmp24 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp27 = tmp25 * tmp26
    tmp28 = tl.full(tmp27.shape, 0.0, tmp27.dtype)
    tmp29 = tl.where(tmp24, tmp27, tmp28)
    tmp30 = tl.full(tmp29.shape, 0.0, tmp29.dtype)
    tmp31 = tl.where(tmp21, tmp29, tmp30)
    tmp32 = 0.0
    tmp33 = tl.where(tmp20, tmp31, tmp32)
    tmp34 = tl.full(tmp33.shape, 0.0, tmp33.dtype)
    tmp35 = tl.where(tmp18, tmp33, tmp34)
    tmp36 = tl.where(tmp18, tmp35, tmp32)
    tmp37 = tl.where(tmp5, tmp16, tmp36)
    tmp38 = tmp0 < tmp17
    tmp39 = tmp20 & tmp38
    tmp40 = (((-256) + y0 + (513*x1) + (262656*x2)) // 512) % 513
    tmp41 = tmp40 < tmp7
    tmp42 = tmp41 & tmp39
    tmp43 = tl.load(in_ptr0 + ((256*((((-256) + y0 + (513*x1) + (262656*x2)) // 262656) % 3)) + ((((-256) + y0 + (513*x1) + (262656*x2)) // 512) % 513)), tmp42 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp44 = tl.load(in_ptr1 + ((256*((((-256) + y0 + (513*x1) + (262656*x2)) // 262656) % 3)) + (((-256) + y0 + (513*x1) + (262656*x2)) % 512)), tmp42 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp45 = tmp43 * tmp44
    tmp46 = tl.full(tmp45.shape, 0.0, tmp45.dtype)
    tmp47 = tl.where(tmp42, tmp45, tmp46)
    tmp48 = tl.full(tmp47.shape, 0.0, tmp47.dtype)
    tmp49 = tl.where(tmp39, tmp47, tmp48)
    tmp50 = tl.where(tmp20, tmp49, tmp32)
    tmp51 = tl.full(tmp50.shape, 0.0, tmp50.dtype)
    tmp52 = tl.where(tmp38, tmp50, tmp51)
    tmp53 = tl.where(tmp38, tmp52, tmp32)
    tmp54 = tl.where(tmp2, tmp37, tmp53)
    tl.store(out_ptr0 + (x3 + (1024*y0)), tmp54, xmask & ymask)
