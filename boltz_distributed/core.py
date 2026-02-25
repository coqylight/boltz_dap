"""
DAP Core - Initialization for Dynamic Axial Parallelism

Adapted from FastFold's fastfold/distributed/core.py
"""

import os
import torch


def ensure_divisibility(numerator, denominator):
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, f'{numerator} is not divisible by {denominator}'


def set_missing_distributed_environ(key, value):
    """Set environment variable if not already set."""
    if key not in os.environ:
        os.environ[str(key)] = str(value)


# Global DAP configuration
_DAP_INITIALIZED = False
_DAP_SIZE = 1
_DAP_RANK = 0
_DAP_GROUP = None


def init_dap(dap_size=None):
    """
    Initialize Dynamic Axial Parallelism.
    
    This can work in two modes:
    1. With ColossalAI (if available) - full tensor parallelism support
    2. Standalone with PyTorch distributed - simpler but still functional
    
    Args:
        dap_size: Number of GPUs to use. If None, uses WORLD_SIZE or 1.
    """
    global _DAP_INITIALIZED, _DAP_SIZE, _DAP_RANK, _DAP_GROUP
    
    if _DAP_INITIALIZED:
        return
    
    # Determine DAP size
    if dap_size is None:
        if 'WORLD_SIZE' in os.environ:
            dap_size = int(os.environ['WORLD_SIZE'])
        else:
            dap_size = 1
    
    # Try to use ColossalAI first (recommended)
    try:
        import colossalai
        from colossalai.context.parallel_mode import ParallelMode
        from colossalai.core import global_context as gpc
        
        colossalai.logging.disable_existing_loggers()
        
        if torch.distributed.is_initialized():
            raise RuntimeError(
                "Use init_dap instead of torch.distributed.init_process_group!"
            )
        
        # Set distributed environ for single device launch
        set_missing_distributed_environ('WORLD_SIZE', 1)
        set_missing_distributed_environ('RANK', 0)
        set_missing_distributed_environ('LOCAL_RANK', 0)
        set_missing_distributed_environ('MASTER_ADDR', "localhost")
        set_missing_distributed_environ('MASTER_PORT', 18417)
        
        colossalai.launch_from_torch(
            config={"parallel": dict(tensor=dict(size=dap_size))}
        )
        
        _DAP_SIZE = gpc.get_world_size(ParallelMode.TENSOR)
        _DAP_RANK = gpc.get_local_rank(ParallelMode.TENSOR)
        _DAP_GROUP = gpc.get_group(ParallelMode.TENSOR)
        _DAP_INITIALIZED = True
        
        print(f"[BoltzDAP] Initialized with ColossalAI, DAP size: {_DAP_SIZE}")
        
    except ImportError:
        # Fallback to pure PyTorch distributed
        print("[BoltzDAP] ColossalAI not found, using PyTorch distributed...")
        
        # Get local rank for GPU assignment
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        # Set CUDA device BEFORE initializing distributed
        torch.cuda.set_device(local_rank)
        
        if not torch.distributed.is_initialized():
            set_missing_distributed_environ('WORLD_SIZE', dap_size)
            set_missing_distributed_environ('RANK', 0)
            set_missing_distributed_environ('LOCAL_RANK', 0)
            set_missing_distributed_environ('MASTER_ADDR', "localhost")
            set_missing_distributed_environ('MASTER_PORT', 18417)
            
            from datetime import timedelta
            nccl_timeout = int(os.environ.get('NCCL_TIMEOUT', 7200))
            torch.distributed.init_process_group(
                backend='nccl',
                init_method='env://',
                timeout=timedelta(seconds=nccl_timeout),  # default 2h — --no_kernels diffusion can take >45 min
            )
        
        _DAP_SIZE = torch.distributed.get_world_size()
        _DAP_RANK = torch.distributed.get_rank()
        _DAP_GROUP = None  # Use default group
        _DAP_INITIALIZED = True
        
        print(f"[BoltzDAP] Initialized with PyTorch distributed, DAP size: {_DAP_SIZE}, rank: {_DAP_RANK}, GPU: {local_rank}")


def get_dap_size():
    """Get the DAP world size."""
    return _DAP_SIZE


def get_dap_rank():
    """Get the DAP local rank."""
    return _DAP_RANK


def get_dap_group():
    """Get the DAP process group."""
    return _DAP_GROUP


def is_dap_initialized():
    """Check if DAP is initialized."""
    return _DAP_INITIALIZED
