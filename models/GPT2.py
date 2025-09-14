from torch import Tensor
import torch.nn as nn

from layers.attention import CausalAttention

"""
Basics of a feed forward network
operates on each token independent of other tokens
cannot reference other tokens or pos info outside of info embedded in current token vector

formally, FF layer is defined as

FFN = Act(XW^{1})W^{2}
where W^{1} in R^{d_{model} x d_{FFN}}
is a linear layer projecting token vectors into higher dim space d_{FFN}
Act is the activation function,
W^{2} in R^{d_{FFN} x d_{model}}
projects expanded token vectors back down to input space d_model

NOTE:
Act(XW^{1})W^{2} is a condensed expression for two sets of linear equatiosn with a non-linearity in between:

FFN = Act(XA^{1} + B^{1})A^{2} + B^{2}
where as and bs are learnable params for calculating output from input X

~= to implicit KV memory to transformer layer,
with upscaling projection generating per-token keys
into FFN working memory

Neurons in FF layers thought to respond to multiple concepts at once

superposition hypo suggests polysemanticity simulates much larger layer
allowing model to understand more feats then params

softmax linear units interp study found that
early, middle and late FF layers likely focus on
diff aspects of language modeling

early: involved w detokenizing inputs into concepts
recognize multi-token words, names, etc
middle: respond to abstract ideas, like num of ppl
final: tend to focus on converting the discrete concepts back to tokens
"""

# FF implementation
# first linear layer projects token vector from input to expanded
# second linear reverses it
# in this implementation, place optional dropout layer after last linear, but some models place it before the last linear

# like attn layer, linear layers in FF will optionally allow biases to be disabled for increased throughput and reduced memory usage


class FeedForward(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        expand_size: int,
        act: nn.Module = nn.GELU,
        drop: float = 0.1,
        bias: bool = True,
    ):
        super().__init__()
        # project input
        self.fc1 = nn.Linear(hidden_size, expand_size, bias=bias)
        # add non-linearity
        self.act = act()
        # project back
        self.fc2 = nn.Linear(expand_size, hidden_size, bias=bias)
        # optional dropout to avoid overfit
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# Transformer block

"""
With FFN defined
now have pieces for transformer layer/block

block sequentially calculates attn and ffn layers w
residual connections and normalization layers in-between

two predominant variants of resid-connect and norm layers:
post and pre norm

post: used in AAYN, BERT
applies norm layer to both Attn/FF layer and residual

Y = Norm(X + Attention(X))
Output = Norm(Y + FFN(Y))

can suffer from grad vanishing as normalization is applied multiple times
can cause grad norm to get exponentially small
hindering model training

using small learning rates and learning rate warmup improves post-norm training

pre-norm applies norm layer to input before passed to layers

Y = X + Attn(Norm(X))
Output = Y + FFN(Norm(Y))

potentially can cause representational collapse
last model layers very similar, contributing little to model capcaity

pre norm transformers can train faster bc of stable grad flow btwn layers
allows higher learn rates, reduces need for warmup

most modern transformer based LLMs use pre-norm or variants
eg: GPT, T5, Cramming BERT, MPT, Falcon all use pre-norm
"""

# Block implementation


class TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        context_size: int,
        expand_size: int,
        attention: nn.Module = CausalAttention,
        act: nn.Module = nn.GELU,
        attn_drop: float = 0.1,
        out_drop: float = 0.1,
        ffn_drop: float = 0.1,
        bias: bool = True,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = attention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            context_size=context_size,
            attn_drop=attn_drop,
            out_drop=out_drop,
            bias=bias,
        )
        self.norm2 = nn.LayerNorm(hidden_size)
        self.ffn = FeedForward(
            hidden_size=hidden_size,
            expand_size=expand_size,
            act=act,
            drop=ffn_drop,
            bias=bias,
        )

    def forward(self, x: Tensor):
        # im using pre-norm here
        x = x + self.attn(self.norm1(x))
        # normalize input then add resid to ff out
        return x + self.ffn(self.norm2(x))
