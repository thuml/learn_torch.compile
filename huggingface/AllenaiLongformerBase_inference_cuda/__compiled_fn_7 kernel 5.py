
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8388608], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_copy_new_zeros_select_scatter_slice_scatter_4', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6303744
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 131328) % 4
    x0 = xindex % 513
    x1 = (xindex // 513) % 256
    x3 = (xindex // 525312)
    x5 = xindex % 131328
    x6 = xindex
    tmp0 = x2
    tmp1 = tl.full([1], 3, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp3 = x0
    tmp4 = tl.full([1], 256, tl.int64)
    tmp5 = tmp3 >= tmp4
    tmp6 = ((656384 + x0 + (513*x1)) // 512) % 513
    tmp7 = tl.full([1], 512, tl.int64)
    tmp8 = tmp6 < tmp7
    tmp9 = tmp8 & tmp5
    tmp10 = tl.load(in_ptr0 + ((512*(((656384 + x5) // 512) % 513)) + (262144*((656384 + x5) // 262656)) + (786432*x3) + (786432*((656384 + x5) // 787968)) + (x5 % 512)), tmp9, eviction_policy='evict_last', other=0.0)
    tmp11 = tl.full(tmp10.shape, 0.0, tmp10.dtype)
    tmp12 = tl.where(tmp9, tmp10, tmp11)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp5, tmp12, tmp13)
    tmp15 = tl.full([1], 3, tl.int64)
    tmp16 = tmp15 < tmp15
    tmp17 = tmp5 & tmp16
    tmp18 = ((787712 + x0 + (513*x1)) // 512) % 513
    tmp19 = tmp18 < tmp7
    tmp20 = tmp19 & tmp17
    tmp21 = tl.load(in_ptr0 + ((262144*(((787712 + x0 + (513*x1)) // 262656) % 3)) + (786432*(((787712 + x0 + (513*x1) + (787968*x3)) // 787968) % 12)) + ((787712 + x0 + (513*x1)) % 262656)), tmp20, eviction_policy='evict_last', other=0.0)
    tmp22 = tl.full(tmp21.shape, 0.0, tmp21.dtype)
    tmp23 = tl.where(tmp20, tmp21, tmp22)
    tmp24 = tl.full(tmp23.shape, 0.0, tmp23.dtype)
    tmp25 = tl.where(tmp17, tmp23, tmp24)
    tmp26 = 0.0
    tmp27 = tl.where(tmp5, tmp25, tmp26)
    tmp28 = tl.full(tmp27.shape, 0.0, tmp27.dtype)
    tmp29 = tl.where(tmp16, tmp27, tmp28)
    tmp30 = tl.where(tmp16, tmp29, tmp26)
    tmp31 = tl.where(tmp5, tmp14, tmp30)
    tmp32 = tmp0 < tmp15
    tmp33 = tmp5 & tmp32
    tmp34 = (((-256) + x0 + (513*x1) + (262656*x2) + (787968*x3)) // 512) % 513
    tmp35 = tmp34 < tmp7
    tmp36 = tmp35 & tmp33
    tmp37 = tl.load(in_ptr0 + ((262144*((((-256) + x0 + (513*x1) + (262656*x2) + (787968*x3)) // 262656) % 36)) + (((-256) + x0 + (513*x1) + (262656*x2) + (787968*x3)) % 262656)), tmp36, other=0.0)
    tmp38 = tl.full(tmp37.shape, 0.0, tmp37.dtype)
    tmp39 = tl.where(tmp36, tmp37, tmp38)
    tmp40 = tl.full(tmp39.shape, 0.0, tmp39.dtype)
    tmp41 = tl.where(tmp33, tmp39, tmp40)
    tmp42 = tl.where(tmp5, tmp41, tmp26)
    tmp43 = tl.full(tmp42.shape, 0.0, tmp42.dtype)
    tmp44 = tl.where(tmp32, tmp42, tmp43)
    tmp45 = tl.where(tmp32, tmp44, tmp26)
    tmp46 = tl.where(tmp2, tmp31, tmp45)
    tl.store(out_ptr0 + (x6), tmp46, None)
