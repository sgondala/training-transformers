import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class NewGeLU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2/math.pi) * (x + 0.044715 * torch.pow(x, 3))))

# This module takes input sequence, attends, and returns final sequence
class CausalSelfAttentionLayer(nn.Module):
    def __init__(self, n_heads, embed_dim, max_seq_len):
        super().__init__()
        self.n_heads = n_heads
        self.embed_dim = embed_dim
        assert embed_dim % n_heads == 0, 'Embedding dimension must be divisible by number of heads'
        # We need KQV for each head but everything in one matrix
        self.kqv_matrices = nn.Linear(embed_dim, 3 * embed_dim)
        # One projection Matrix
        self.proj_matrix = nn.Linear(embed_dim, embed_dim)
        self.register_buffer('mask', 
            torch.tril(torch.ones(max_seq_len, max_seq_len)).view(1,1,max_seq_len,max_seq_len))
        self.attn_dropout = nn.Dropout(0.5)
        self.proj_dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # X should be a tensor of shape (batch, seq len, embed dim)
        batch_size, seq_len, embed_dim = x.shape
        kqv_out = self.kqv_matrices(x) # (batch, seq len, 3 * embed dim)
        q, k, v = kqv_out.split(self.embed_dim, dim=2) # Each is (batch, seq len, embed dim)
        # Need to split into individual heads before we do matmul
        q = q.view(batch_size, seq_len, self.n_heads, self.embed_dim // self.n_heads).transpose(1,2) # batch, n_heads, seq_len, dim_head
        k = k.view(batch_size, seq_len, self.n_heads, self.embed_dim // self.n_heads).transpose(1,2)
        v = v.view(batch_size, seq_len, self.n_heads, self.embed_dim // self.n_heads).transpose(1,2)
        
        # Now we can do the matmul
        att = torch.matmul(q, k.transpose(-1, -2)) # batch, n_heads, seq_len, seq_len
        att = att * 1/math.sqrt(self.embed_dim // self.n_heads) # For normalization
        att = att.masked_fill(self.mask[:,:,:seq_len,:seq_len] == 0, float('-inf')) # Masking future tokens
        att = F.softmax(att, dim=-1) # batch, n_heads, seq_len, seq_len
        att = self.attn_dropout(att) # TODO: Why here and not after V?

        y = torch.matmul(att, v) # batch, n_heads, seq_len, dim_head
        y = y.transpose(1,2).contiguous().reshape(batch_size, seq_len, embed_dim) # batch, seq_len, embed_dim

        y = self.proj_matrix(y)
        y = self.proj_dropout(y)
        return y


class Layer(nn.Module):
    def __init__(self, n_heads, embed_dim, max_seq_len):
       super().__init__()
       self.ln1 = nn.LayerNorm(embed_dim)
       self.attn_layer = CausalSelfAttentionLayer(n_heads, embed_dim, max_seq_len)
       self.ln2 = nn.LayerNorm(embed_dim)
       self.mlp_layers = nn.ModuleDict(dict(
            fc1 = nn.Linear(embed_dim, 4 * embed_dim),
            act = NewGeLU(),
            fc2 = nn.Linear(4 * embed_dim, embed_dim),
            dropout = nn.Dropout(0.5),
       ))
       self.mlp_chain = lambda x : \
            self.mlp_layers['dropout'](
                    self.mlp_layers['fc2'](
                        self.mlp_layers['act'](
                            self.mlp_layers['fc1'](x))))
    
    def forward(self, x):
        # LayerNorm moved to input of each sublayer in GPT-2
        x = x + self.attn_layer(self.ln1(x))
        x = x + self.mlp_chain(self.ln2(x))
        return x
        

class GPT(nn.Module):
    def __init__(self, 
            vocab_size, 
            embed_dim,
            max_seq_len,
            n_layers, 
            n_heads):
        super().__init__()
        self.vocab_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(max_seq_len, embed_dim)
        self.embed_dropout = nn.Dropout(0.5)
        self.max_seq_len = max_seq_len
        self.layers = nn.ModuleList(
            [Layer(n_heads, embed_dim, max_seq_len) for _ in range(n_layers)]
        )
        # Final Layer norm added after final layer
        self.final_layer_norm = nn.LayerNorm(embed_dim)
        # One head to get back to vocab size
        self.lm_head = nn.Linear(embed_dim, vocab_size)
    
    def forward(self, x, targets=None):
        # X is a tensor of shape (batch, seq len)
        device = x.device
        batch_size, seq_len = x.shape
        assert seq_len <= self.max_seq_len, 'Sequence length must be less than max_seq_len'
        pos = torch.arange(seq_len, device=device).unsqueeze(0) # (1, seq_len)

        tok_emb = self.vocab_embed(x) # (batch, seq_len, embed_dim)
        pos_emb = self.pos_embed(pos) # (1, seq_len, embed_dim)

        vectors = self.embed_dropout(tok_emb + pos_emb) # (batch, seq_len, embed_dim)

        for layer in self.layers:
            vectors = layer(vectors)
        
        vectors = self.final_layer_norm(vectors)
        logits = self.lm_head(vectors)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        
        return logits, loss

    
if __name__ == '__main__':
    gpt = GPT(100, 512, 512, 12, 8)
    x = torch.randint(0, 100, (1, 512))
    logits, loss = gpt(x)
    print(logits.shape)
    print(loss)
