
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_27', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 512
    x1 = (xindex // 512)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (8192*x1)), None)
    tmp1 = tl.load(in_ptr0 + (512 + x0 + (8192*x1)), None)
    tmp3 = tl.load(in_ptr0 + (1024 + x0 + (8192*x1)), None)
    tmp5 = tl.load(in_ptr0 + (1536 + x0 + (8192*x1)), None)
    tmp7 = tl.load(in_ptr0 + (2048 + x0 + (8192*x1)), None)
    tmp9 = tl.load(in_ptr0 + (2560 + x0 + (8192*x1)), None)
    tmp11 = tl.load(in_ptr0 + (3072 + x0 + (8192*x1)), None)
    tmp13 = tl.load(in_ptr0 + (3584 + x0 + (8192*x1)), None)
    tmp15 = tl.load(in_ptr0 + (4096 + x0 + (8192*x1)), None)
    tmp17 = tl.load(in_ptr0 + (4608 + x0 + (8192*x1)), None)
    tmp19 = tl.load(in_ptr0 + (5120 + x0 + (8192*x1)), None)
    tmp21 = tl.load(in_ptr0 + (5632 + x0 + (8192*x1)), None)
    tmp23 = tl.load(in_ptr0 + (6144 + x0 + (8192*x1)), None)
    tmp25 = tl.load(in_ptr0 + (6656 + x0 + (8192*x1)), None)
    tmp27 = tl.load(in_ptr0 + (7168 + x0 + (8192*x1)), None)
    tmp29 = tl.load(in_ptr0 + (7680 + x0 + (8192*x1)), None)
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp8 = tmp7 + tmp6
    tmp10 = tmp9 + tmp8
    tmp12 = tmp11 + tmp10
    tmp14 = tmp13 + tmp12
    tmp16 = tmp15 + tmp14
    tmp18 = tmp17 + tmp16
    tmp20 = tmp19 + tmp18
    tmp22 = tmp21 + tmp20
    tmp24 = tmp23 + tmp22
    tmp26 = tmp25 + tmp24
    tmp28 = tmp27 + tmp26
    tmp30 = tmp29 + tmp28
    tmp31 = 0.0625
    tmp32 = tmp30 * tmp31
    tl.store(out_ptr0 + (x2), tmp32, None)
