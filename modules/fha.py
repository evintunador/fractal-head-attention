import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, List
import math

from modules.logging import LoggingModule, log_io

def precompute_freqs_cis(dim: int, end: int, theta: float = 10_000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

class FHA(LoggingModule): 
    def __init__(
        self, 
        dim: int,
        head_dim: int,
        num_q_heads: int,
        num_kv_heads: int,
        max_batch_size: int,
        max_seq_len: int,
        fractal_split: int = 2, # split factor for each level of smaller heads
        dropout_rate: float = 0.1,
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super().__init__()
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_q_heads if num_kv_heads is None else num_kv_heads
        assert num_q_heads % num_kv_heads == 0, f'num_q_heads must be divisible by num_kv_heads'
        self.head_dim = dim // num_q_heads if head_dim is None else head_dim
        self.dropout_rate = dropout_rate

        self.Wq = nn.Linear(dim, num_q_heads * head_dim, bias=False)
        self.Wk = nn.Linear(dim, self.num_kv_heads * head_dim, bias=False)
        self.Wv = nn.Linear(dim, self.num_kv_heads * head_dim, bias=False)
        self.Wo = nn.Linear(num_q_heads * head_dim, dim, bias=False)

        self.cache_k = torch.zeros(
            (max_batch_size, max_seq_len, num_kv_heads, head_dim),
            requires_grad = False).to(device)
        self.cache_v = torch.zeros(
            (max_batch_size, max_seq_len, num_kv_heads, head_dim),
            requires_grad = False).to(device)

        self.fractal_split = fractal_split
        assert head_dim // (fractal_split ** (num_q_heads - 1)) >= 2, \
        'fractal heads are too small.\neither raise head_dim or fractal_delay or lower fractal_split or num_q_heads'
        # Calculate the total number of fractal heads
        self.num_fractal_heads = sum(self.fractal_split ** i for i in range(self.num_q_heads))
        # Preallocate a tensor for holding all fractalized heads
        self.preallocated_tensor = nn.Parameter(torch.zeros(1, max_seq_len, self.num_fractal_heads, head_dim),requires_grad=False)
    
    @log_io
    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
        cache_len: int = None,
        training: bool = False,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        xq, xk, xv = self.Wq(x), self.Wk(x), self.Wv(x)

        xq = xq.view(batch_size, seq_len, self.num_q_heads, self.head_dim)
        xk = xk.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        xq, xk = self.apply_rotary_emb(xq, xk, freqs_cis)

        if cache_len is not None: # if we're performing inference and using kv caching. it'll init at 0
            self.cache_k = self.cache_k.to(xq)
            self.cache_v = self.cache_v.to(xq)

            self.cache_k[:batch_size, cache_len : cache_len + seq_len] = xk
            self.cache_v[:batch_size, cache_len : cache_len + seq_len] = xv

            keys = self.cache_k[:batch_size, : cache_len + seq_len]
            values = self.cache_v[:batch_size, : cache_len + seq_len]
        else: 
            # if we're training, do full sequence length
            keys, values = xk, xv
        queries = xq # for sake of keeping the naming scheme consistent

        # adjusts keys and values to match the query heads count.
        if self.num_kv_heads != self.num_q_heads:
            keys, values = self.match_headcount(keys, values) # (bs, cache_len + seq_len, num_q_heads, head_dim)

        q = self.fractalize_heads_tensor(queries).transpose(1,2)
        k = self.fractalize_heads_tensor(keys).transpose(1,2)
        v = self.fractalize_heads_tensor(values).transpose(1,2)
        
        logits = self.attend(q, k, training)
        
        if mask is not None:
            logits = logits + mask  # (bs, num_fractal_heads, seq_len, cache_len + seq_len)
            
        scores = self.calc_output(logits, v, training)
        scores = self.desparsify_scores(scores)
        
        output = self.Wo(scores)
        if training: output = F.dropout(output, self.dropout_rate)
        
        return output
    
    @log_io
    def apply_rotary_emb(
        self,
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cis: torch.Tensor,
    ) -> (torch.Tensor, torch.Tensor):
        xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
        xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
        freqs_cis = self.reshape_for_broadcast(freqs_cis.to(xq.device), xq_)
        xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
        xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
        return xq_out.type_as(xq), xk_out.type_as(xk)

    @log_io
    def reshape_for_broadcast(
        self,
        freqs_cis: torch.Tensor, 
        x: torch.Tensor
    ) -> torch.Tensor:
        ndim = x.ndim
        assert 0 <= 1 < ndim
        assert freqs_cis.shape == (x.shape[1], x.shape[-1]), \
        f'freqs_cis.shape {freqs_cis.shape} != (x.shape[1], x.shape[-1]) {(x.shape[1], x.shape[-1])}'
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return freqs_cis.view(*shape)

    @log_io
    def match_headcount(
        self, 
        keys: torch.Tensor, 
        values: torch.Tensor
    ) -> (torch.Tensor, torch.Tensor):
        keys = torch.repeat_interleave(keys, self.num_q_heads // self.num_kv_heads, dim=2)
        values = torch.repeat_interleave(values, self.num_q_heads // self.num_kv_heads, dim=2)
        return keys, values

    @log_io
    def fractalize_heads_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """
        the key function of this experiment
        iterates through each of the original q, k, & v heads and splits them into smaller heads according to fractal_split
        then places them into our preallocated tensor full of zeros
        
        although these zeros take up a bit of memory and flops, doing it this way allows us to treat 
        most of the remaining operations as if we were doing regular multi-head-attention 

        x: Tensor of shape (batch_size, seq_len, num_q_heads, head_dim)
        expanded_tensor: Tensor of shape (batch_size, seq_len, num_fractal_heads, head_dim) where most entries are 0
        """
        batch_size, seq_len, _, _ = x.shape
        # Expand the preallocated tensor to the current batch size
        expanded_tensor = self.preallocated_tensor.repeat(batch_size, 1, 1, 1)[:,:seq_len,...]
        
        head_index = 0
        for i in range(self.num_q_heads):
            split_size = self.fractal_split ** i
            new_head_dim = self.head_dim // split_size
            
            for j in range(split_size):
                expanded_tensor[:, :, head_index, :new_head_dim] = x[:, :, i, j*new_head_dim:(j+1)*new_head_dim]
                head_index += 1
        
        return expanded_tensor

    @log_io
    def attend(
        self, 
        queries: 
        torch.Tensor, 
        keys: torch.Tensor, 
        training: bool
    ) -> torch.Tensor:
        return torch.matmul(queries, keys.transpose(2, 3)) * (self.head_dim ** -0.5)
    
    @log_io
    def calc_output(
        self, 
        logits: torch.Tensor, 
        values: torch.Tensor, 
        training: bool
    ) -> torch.Tensor:
        batch_size, _, seq_len, _ = logits.shape
        scores = F.softmax(logits, dim=-1)
        if training: scores = F.dropout(scores, self.dropout_rate)
        output = scores @ values # [batch_size, n_heads, seq_len, head_dim]
        return output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1) # [batch_size, seq_len, n_heads * head_dim]

    @log_io
    def desparsify_scores(
        self,
        scores: torch.Tensor
    ) -> torch.Tensor:
        """
        turns scores from a very sparse tensor of mostly zeros shape (batch_size, seq_len, head_dim * num_fractal_heads)
        back into a regular (batch_size, seq_len, head_dim * num_q_heads) that's compatible with self.Wo
        """
        # so here was my scratch work when making the below function attempting to remove all the zeros from scores
        # 32, 16, 16, 8, 8, 8, 8, 4, 4, 4, 4, 4, 4, 4, 4 <- each head size
        # 32//(2**f(i)) <- i'll calculate head sizes using that, but idk what f(i) should be
        # 0,  1,  2,  3, 4, 5, 6, 7, 8, 9,10,11,12,13,14 <- i
        # 0,  1,  1,  2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3 <- f(i)
        # looks like that should be math.floor(math.log2(i + 1))
        
        scores = torch.concat(
            [scores[:,:, i * self.head_dim:i * self.head_dim + (32 // (2 ** math.floor(math.log2(i + 1))))] for i in range(self.num_fractal_heads)], 
            dim=2
        )
        return scores