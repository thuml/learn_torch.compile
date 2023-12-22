
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_19', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3010560
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 3136) % 120
    x2 = (xindex // 376320)
    x3 = xindex % 376320
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 100, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.full([1], 80, tl.int64)
    tmp6 = tmp0 < tmp5
    tmp7 = tmp6 & tmp4
    tmp8 = tl.full([1], 60, tl.int64)
    tmp9 = tmp0 < tmp8
    tmp10 = tmp9 & tmp7
    tmp11 = tl.full([1], 40, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = tmp12 & tmp10
    tmp14 = tl.load(in_ptr0 + (802816 + x3 + (928256*x2)), tmp13, other=0.0)
    tmp15 = tl.full(tmp14.shape, 0.0, tmp14.dtype)
    tmp16 = tl.where(tmp13, tmp14, tmp15)
    tmp17 = tmp0 >= tmp11
    tmp18 = tmp17 & tmp10
    tmp19 = tl.load(in_ptr1 + (677376 + x3 + (865536*x2)), tmp18, other=0.0)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = tl.where(tmp12, tmp16, tmp21)
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp10, tmp22, tmp23)
    tmp25 = tmp0 >= tmp8
    tmp26 = tmp25 & tmp7
    tmp27 = tl.load(in_ptr2 + (614656 + x3 + (865536*x2)), tmp26, other=0.0)
    tmp28 = tl.full(tmp27.shape, 0.0, tmp27.dtype)
    tmp29 = tl.where(tmp26, tmp27, tmp28)
    tmp30 = tl.where(tmp9, tmp24, tmp29)
    tmp31 = tl.full(tmp30.shape, 0.0, tmp30.dtype)
    tmp32 = tl.where(tmp7, tmp30, tmp31)
    tmp33 = tmp0 >= tmp5
    tmp34 = tmp33 & tmp4
    tmp35 = tl.load(in_ptr3 + (551936 + x3 + (865536*x2)), tmp34, other=0.0)
    tmp36 = tl.full(tmp35.shape, 0.0, tmp35.dtype)
    tmp37 = tl.where(tmp34, tmp35, tmp36)
    tmp38 = tl.where(tmp6, tmp32, tmp37)
    tmp39 = tl.full(tmp38.shape, 0.0, tmp38.dtype)
    tmp40 = tl.where(tmp4, tmp38, tmp39)
    tmp41 = tmp0 >= tmp3
    tmp42 = tl.full([1], 120, tl.int64)
    tmp43 = tmp0 < tmp42
    tmp44 = tl.load(in_ptr4 + (489216 + x3 + (865536*x2)), tmp41, other=0.0)
    tmp45 = tl.full(tmp44.shape, 0.0, tmp44.dtype)
    tmp46 = tl.where(tmp41, tmp44, tmp45)
    tmp47 = tl.where(tmp4, tmp40, tmp46)
    tl.store(out_ptr0 + (x3 + (1179136*x2)), tmp47, None)
