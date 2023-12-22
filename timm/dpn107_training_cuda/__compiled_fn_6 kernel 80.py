
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_slice_backward_79', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1705984
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 196) % 1088
    x2 = (xindex // 213248)
    x3 = xindex % 213248
    x4 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 1024, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + ((-163072) + x3 + (75264*x2)), tmp2, other=0.0)
    tmp4 = tl.load(in_ptr1 + (37632 + x3 + (263424*x2)), tmp2, other=0.0)
    tmp5 = tmp3 + tmp4
    tmp6 = tl.load(in_ptr2 + (37632 + x3 + (250880*x2)), tmp2, other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp2, tmp7, tmp8)
    tmp10 = 0.0
    tmp11 = tl.where(tmp2, tmp9, tmp10)
    tmp12 = tmp0 < tmp1
    tmp13 = tl.load(in_ptr3 + (x3 + (200704*x2)), tmp12, other=0.0)
    tmp14 = tl.load(in_ptr1 + (x3 + (263424*x2)), tmp12, other=0.0)
    tmp15 = tmp13 + tmp14
    tmp16 = tl.load(in_ptr2 + (x3 + (250880*x2)), tmp12, other=0.0)
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full(tmp17.shape, 0.0, tmp17.dtype)
    tmp19 = tl.where(tmp12, tmp17, tmp18)
    tmp20 = tl.where(tmp12, tmp19, tmp10)
    tmp21 = tmp11 + tmp20
    tl.store(out_ptr0 + (x4), tmp21, None)
