import torch
import torch.nn as nn
import typing
from fvcore.nn import FlopCountAnalysis, flop_count_str
from typing import Any, Callable, List, Optional, Union, Dict
from numbers import Number

Handle = Callable[[List[Any], List[Any]], Union[typing.Counter[str], Number]]

def get_shape(val: Any) -> Optional[List[int]]:
    if val.isCompleteTensor():
        return val.type().sizes()
    else:
        return None

def baddbmm_flop_jit(inputs: List[Any], outputs: List[Any]) -> Number:
    """
    Count FLOPs for torch.baddbmm.
    Performs a batched matrix multiply and an add.
    FLOPs = B * M * N * P
    """
    input_shapes = [get_shape(v) for v in inputs[1:3]]
    # batch1: [B, N, M]
    # batch2: [B, M, P]
    assert len(input_shapes[0]) == 3 and len(input_shapes[1]) == 3, input_shapes
    B, N, M = input_shapes[0]
    P = input_shapes[1][2]
    flops =  B * N * M * P
    return flops

def scaled_dot_product_attention_flop_jit(inputs: List[Any], outputs: List[Any]) -> Number:
    """
    Count FLOPs for torch.nn.functional.scaled_dot_product_attention,
    excluding softmax and scaling.

    Operation: output = (Q @ K.T) @ V

    Shapes:
    - Q: (B, H, N, Dk)
    - K: (B, H, S, Dk)
    - V: (B, H, S, Dv)

    FLOPs = B * H * N * S * (Dk + Dv)
    """

    q_shape = get_shape(inputs[0])
    k_shape = get_shape(inputs[1])
    v_shape = get_shape(inputs[2])

    if len(q_shape) == 3:
        B, N, Dk = q_shape
        H = 1
        S = k_shape[1]
        Dv = v_shape[2]
    elif len(q_shape) == 4:
        B, H, N, Dk = q_shape
        S = k_shape[2]
        Dv = v_shape[3]
    else:
        raise ValueError(f"Unsupported query shape {q_shape}")

    flops = B * H * N * S * (Dk + Dv)
    return flops

class ModelProfiler:
    def __init__(self, model, input_sample):
        self.model = model
        self.input_sample = input_sample
        self.device = next(model.parameters()).device

    def count_params(self) -> Dict[str, int]:
        total_params = sum(p.numel() for p in self.model.parameters())
        ignored_params = 0
        
        def process_block(block):
            nonlocal ignored_params
            if hasattr(block, 'moe_num_experts') and block.moe_num_experts > 1:
                top_k = getattr(block, 'moe_top_k', 1)
                num_experts = block.moe_num_experts
                
                expert_0 = block.experts[0]
                expert_params = sum(p.numel() for p in expert_0.parameters())
                
                experts_skipped = num_experts - top_k
                ignored_params += (experts_skipped * expert_params)

        for module in self.model.modules():
            if hasattr(module, 'experts') and isinstance(module.experts, nn.ModuleList):
                process_block(module)
                
        active_params = total_params - ignored_params
        return {
            "total_params": total_params, 
            "active_params": active_params,
            "utilization": active_params / total_params
        }

    def count_flops(self) -> Dict[str, Any]:
        handlers = {
            "aten::scaled_dot_product_attention": scaled_dot_product_attention_flop_jit,
            "aten::baddbmm": baddbmm_flop_jit
        }
        
        flop = FlopCountAnalysis(self.model, self.input_sample)
        flop.set_op_handle(**handlers)
        return {
            "total_flops": flop.total(), 
            "flops_str": flop_count_str(flop)
        }

    def measure_latency(self, num_warmup=10, num_steps=100) -> Dict[str, float]:
        self.model.eval()
        
        # warmup
        with torch.no_grad():
            for _ in range(num_warmup):
                if isinstance(self.input_sample, tuple):
                    self.model(*self.input_sample)
                else:
                    self.model(self.input_sample)
        
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        torch.cuda.synchronize()
        start.record()
        
        with torch.no_grad():
            for _ in range(num_steps):
                if isinstance(self.input_sample, tuple):
                    self.model(*self.input_sample)
                else:
                    self.model(self.input_sample)
                    
        end.record()
        torch.cuda.synchronize()
        
        avg_ms = start.elapsed_time(end) / num_steps
        return {"latency_ms": avg_ms}