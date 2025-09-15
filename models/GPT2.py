import math
import torch
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


# Vocabulary Embeddings
"""
how a transformer model turns input tokens
in NLP, words, phrases, subwords converted to discrete integers, into a continuous token vector

as model is trained, vector will capture semantic info about each token

choosing correct size for vocab important
both for downstream perf and comp efficiency

tradeoff when inceasing vocab size btwn
training diff due to more tokens
representational ease bc compressing more info into a fixed number of tokens

was found in Cramming that increasing BERTs vocab improves downstream perf until plateauing at original vocab size of 32k tokens

computational eff: karpathy found increasing nanoGPTs vocab from 50257 to 50304 led to ~25% inc in training speed, less so for larger models but still substantial

for efficiency, a power of 2 and multiple of 8,64 and/or 128 important for hardware tiling
which provides speedup, sharding across multiple accels

torch implements its embeddings as lookup table of shape
input_dim, embedding_dim

each row contains embedding vector for specific input token
output of embedding layer is token vector passed through model's transformer layers

define vocab embedding by passing in vocab size to
nn.Embedding
"""

"""
# embeddings of shape vocab size, embedding size C
# input shape (B,S), out shape (B,S,C)
vocab_embed = nn.Embedding(vocab_size, hidden_size)
"""

# positional encodings and embeddings
"""
outside of causal masking
attn treats all positions equally
this is rectified by adding positional embedding/encoding vectors to token vectors

original transformer used sinusoidal positional encodings
"""


class PositionalEncoding(nn.Module):
    def __init__(self, context_size: int, hidden_size: int):
        super().__init__()
        # max seq len by embedding dim
        pe = torch.zeros(context_size, hidden_size, dtype=torch.float)
        # pre-populate
        position = torch.arange(context_size).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_size, 2) * (-math.log(10000) / hidden_size)
        )
        # even uses sin, odd cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: Tensor):
        return self.pe[:, : x.shape[1], :]


# weight tying
"""
last layer is prediction head
converts token vectors processed by attn, FF
into token predictions
this is a prob distribution over all possible tokens in vocab

model head implemented as single linear layer
with optional preceding normalization layer
like all linear layers here, bias optional

# converts input token vectors of shape (B, S, C) to probability
# distribution of shape batch, sequence length, vocabulary size (B, S, VS)
head = nn.Linear(hidden_size, vocab_size, bias=head_bias)

since vocab embedding and prediction head
share input and output dimensions
ph has same weight shape as vocab embed
lead to weight tying
practice of setting vocab embedding to share same set of weights as prediction head

assumes enough similarity btwn
creating token embed
predicting tokens
thf shared weights can learn a repr for both tasks
in practice it works

has 2 main benefits
reduces number of parameters in the model
can also improve model convergence
w/o, vocab embed only updates for curr batch tokens
pred head receives update across vocab
bc zipfs law, tokens in long tails have far fewer updates
bc halving effect, fewer updates for combined embedding
also updates each step bc of pred head

potentially significant downside of perf hit relative to model trained w/o tying

mechanically easy bc just set head weight to vocab weight after head is defined

if tie_weights:
    self.head.weight = self.vocab_embed.weight

if pred head has bias, not tied, as embedding has no bias term
"""


class GPT2(nn.Module):
    def __init__(
        self,
        num_layers: int,
        vocab_size: int,
        hidden_size: int,
        num_heads: int,
        context_size: int,
        expand_size: int,
        attention: nn.Module = CausalAttention,
        act: nn.Module = nn.GELU,
        embed_drop: float = 0.1,
        attn_drop: float = 0.1,
        out_drop: float = 0.1,
        ffn_drop: float = 0.1,
        head_norm: bool = True,
        tie_weights: bool = True,
        head_bias: bool = True,
        bias: bool = True,
    ):
        # init vocab and pos embeddings
        self.vocab_embed = nn.Embedding(vocab_size, hidden_size)
        self.pos_embed = nn.Embedding(context_size, hidden_size)
        self.embed_drop = nn.Dropout(embed_drop)

        # init num layers of transformer layers
        self.tfm_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    context_size=context_size,
                    expand_size=expand_size,
                    attention=attention,
                    act=act,
                    bias=bias,
                    attn_drop=attn_drop,
                    out_drop=out_drop,
                    ffn_drop=ffn_drop,
                )
                for _ in range(num_layers)
            ]
        )

        # optional prehead norming
        if head_norm:
            self.head_norm = nn.LayerNorm(hidden_size)
        else:
            self.head_norm = nn.Identity()

        self.head = nn.Linear(hidden_size, vocab_size, bias=head_bias)

        if tie_weights:
            self.head_weight = self.vocab_embed.weight

        # precreate pos indices
        pos = torch.arange(0, context_size, dtype=torch.long)
        self.register_buffer("pos", pos, persistent=False)

        self.apply(self._init_weights)

    def forward(self, x: Tensor):
        tokens = self.vocab_embed(x)
        pos = self.pos_embed(self.pos[: x.shape[1]])

        x = self.embed_drop(tokens + pos)

        for block in self.tfm_blocks:
            x = block(x)

        x = self.head_norm(x)

        return self.head(x)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            if module._get_name() == "fc2":
                # gpt-2 style ffn init
                torch.nn.init.normal_(
                    module.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.num_layers)
                )
            else:
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


class GPT2ForCausalLM(GPT2):
    def __init__(self, loss_fn: nn.Module = nn.CrossEntropyLoss(), **kwargs):
        super().__init__(**kwargs)
        self.loss_fn = loss_fn

    def forward(self, x: Tensor):
        inputs = x[:, :-1]
        labels = x[:, 1:]

        logits = super().forward(inputs)
        loss = self.loss_fn(logits.view(-1, logits.shape[-1]), labels.view(-1))

        return {"logits": logits, "loss": loss}
