
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__adaptive_avg_pool2d_convolution_max_pool2d_with_indices_relu_13', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 36864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 6
    x1 = (xindex // 6) % 6
    x2 = (xindex // 36)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + ((2*x0) + (26*x1) + (169*x2)), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + (2*x0) + (26*x1) + (169*x2)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (2 + (2*x0) + (26*x1) + (169*x2)), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (13 + (2*x0) + (26*x1) + (169*x2)), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (14 + (2*x0) + (26*x1) + (169*x2)), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (15 + (2*x0) + (26*x1) + (169*x2)), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (26 + (2*x0) + (26*x1) + (169*x2)), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (27 + (2*x0) + (26*x1) + (169*x2)), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (28 + (2*x0) + (26*x1) + (169*x2)), None, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp8 = triton_helpers.maximum(tmp7, tmp6)
    tmp10 = triton_helpers.maximum(tmp9, tmp8)
    tmp12 = triton_helpers.maximum(tmp11, tmp10)
    tmp14 = triton_helpers.maximum(tmp13, tmp12)
    tmp16 = triton_helpers.maximum(tmp15, tmp14)
    tl.store(in_out_ptr0 + (x3), tmp16, None)
