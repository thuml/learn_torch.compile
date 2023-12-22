
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2097152], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_copy_full_like_slice_scatter_where_6', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1575936
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 513) % 256
    x0 = xindex % 513
    x2 = (xindex // 131328)
    x3 = xindex % 131328
    x4 = xindex
    tmp50 = tl.load(in_ptr1 + (x3 + (525312*x2)), xmask)
    tmp0 = x1
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = x0
    tmp4 = tmp3 >= tmp1
    tmp5 = tl.full([1], 256, tl.int64)
    tmp6 = tmp3 < tmp5
    tmp7 = tmp4 & tmp6
    tmp8 = tmp7 & tmp2
    tmp9 = (((-256) + x0 + (513*x1) + (787968*x2)) // 512) % 513
    tmp10 = tl.full([1], 512, tl.int64)
    tmp11 = tmp9 < tmp10
    tmp12 = tmp11 & tmp8
    tmp13 = tl.load(in_ptr0 + ((262144*((((-256) + x0 + (513*x1) + (787968*x2)) // 262656) % 36)) + (((-256) + x0 + (513*x1) + (787968*x2)) % 262656)), tmp12 & xmask, other=0.0)
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp12, tmp13, tmp14)
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp8, tmp15, tmp16)
    tmp18 = tl.full([1], 0, tl.int64)
    tmp19 = tmp18 >= tmp1
    tmp20 = tmp19 & tmp2
    tmp21 = tmp6 & tmp20
    tmp22 = (((-131584) + x0 + (513*x1) + (787968*x2)) // 512) % 513
    tmp23 = tmp22 < tmp10
    tmp24 = tmp23 & tmp21
    tmp25 = tl.load(in_ptr0 + ((512*((((-131584) + x3 + (787968*x2)) // 512) % 513)) + (262144*((((-131584) + x3 + (787968*x2)) // 262656) % 36)) + (x3 % 512)), tmp24 & xmask, other=0.0)
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp24, tmp25, tmp26)
    tmp28 = tl.full(tmp27.shape, 0.0, tmp27.dtype)
    tmp29 = tl.where(tmp21, tmp27, tmp28)
    tmp30 = tl.load(in_ptr1 + (x3 + (525312*x2)), tmp20 & xmask, other=0.0)
    tmp31 = tl.where(tmp6, tmp29, tmp30)
    tmp32 = tl.full(tmp31.shape, 0.0, tmp31.dtype)
    tmp33 = tl.where(tmp20, tmp31, tmp32)
    tmp34 = tl.load(in_ptr1 + (x3 + (525312*x2)), tmp2 & xmask, other=0.0)
    tmp35 = tl.where(tmp19, tmp33, tmp34)
    tmp36 = tl.where(tmp7, tmp17, tmp35)
    tmp37 = tl.full(tmp36.shape, 0.0, tmp36.dtype)
    tmp38 = tl.where(tmp2, tmp36, tmp37)
    tmp39 = tmp6 & tmp19
    tmp40 = tmp23 & tmp39
    tmp41 = tl.load(in_ptr0 + ((512*((((-131584) + x3 + (787968*x2)) // 512) % 513)) + (262144*((((-131584) + x3 + (787968*x2)) // 262656) % 36)) + (x3 % 512)), tmp40 & xmask, other=0.0)
    tmp42 = tl.full(tmp41.shape, 0.0, tmp41.dtype)
    tmp43 = tl.where(tmp40, tmp41, tmp42)
    tmp44 = tl.full(tmp43.shape, 0.0, tmp43.dtype)
    tmp45 = tl.where(tmp39, tmp43, tmp44)
    tmp46 = tl.load(in_ptr1 + (x3 + (525312*x2)), tmp19 & xmask, other=0.0)
    tmp47 = tl.where(tmp6, tmp45, tmp46)
    tmp48 = tl.full(tmp47.shape, 0.0, tmp47.dtype)
    tmp49 = tl.where(tmp19, tmp47, tmp48)
    tmp51 = tl.where(tmp19, tmp49, tmp50)
    tmp52 = tl.where(tmp2, tmp38, tmp51)
    tmp53 = tl.full([1], 257, tl.int64)
    tmp54 = tmp3 < tmp53
    tmp55 = tl.load(in_ptr2 + (x0 + (257*x1)), tmp54 & xmask, eviction_policy='evict_last', other=0.0)
    tmp56 = (tmp55 != 0)
    tmp57 = tl.full([1], 0, tl.int32)
    tmp58 = tmp57 == tmp57
    tmp59 = tmp19 & tmp54
    tmp60 = tmp6 & tmp59
    tmp61 = tmp23 & tmp60
    tmp62 = tl.load(in_ptr0 + ((512*((((-131584) + x3 + (787968*x2)) // 512) % 513)) + (262144*((((-131584) + x3 + (787968*x2)) // 262656) % 36)) + (x3 % 512)), tmp61 & xmask, other=0.0)
    tmp63 = tl.full(tmp62.shape, 0.0, tmp62.dtype)
    tmp64 = tl.where(tmp61, tmp62, tmp63)
    tmp65 = tl.full(tmp64.shape, 0.0, tmp64.dtype)
    tmp66 = tl.where(tmp60, tmp64, tmp65)
    tmp67 = tl.load(in_ptr1 + (x3 + (525312*x2)), tmp59 & xmask, other=0.0)
    tmp68 = tl.where(tmp6, tmp66, tmp67)
    tmp69 = tl.full(tmp68.shape, 0.0, tmp68.dtype)
    tmp70 = tl.where(tmp59, tmp68, tmp69)
    tmp71 = tl.load(in_ptr1 + (x3 + (525312*x2)), tmp54 & xmask, other=0.0)
    tmp72 = tl.where(tmp19, tmp70, tmp71)
    tmp73 = tl.where(tmp58, tmp52, tmp72)
    tmp74 = float("-inf")
    tmp75 = tl.where(tmp56, tmp74, tmp73)
    tmp76 = tl.full(tmp75.shape, 0.0, tmp75.dtype)
    tmp77 = tl.where(tmp54, tmp75, tmp76)
    tmp78 = tl.where(tmp58, tmp52, tmp51)
    tmp79 = tl.where(tmp54, tmp77, tmp78)
    tl.store(out_ptr0 + (x4), tmp52, xmask)
    tl.store(out_ptr1 + (x4), tmp79, xmask)
