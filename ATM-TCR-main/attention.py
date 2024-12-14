#original code
# import torch
# import torch.nn as nn

# class Net(nn.Module):
#     def __init__(self, embedding, args):
#         super(Net, self).__init__()

#         # Embedding Layer
#         self.num_amino = len(embedding)
#         self.embedding_dim = len(embedding[0])
#         self.embedding = nn.Embedding(self.num_amino, self.embedding_dim, padding_idx=self.num_amino-1)
#         if not (args.blosum is None or args.blosum.lower() == 'none'):
#             self.embedding = self.embedding.from_pretrained(torch.FloatTensor(embedding), freeze=False)

#         self.attn_tcr = nn.MultiheadAttention(embed_dim = self.embedding_dim, num_heads = args.heads)
#         self.attn_pep = nn.MultiheadAttention(embed_dim = self.embedding_dim, num_heads = args.heads)

#         # Dense Layer
#         self.size_hidden1_dense = 2 * args.lin_size
#         self.size_hidden2_dense = 1 * args.lin_size
#         self.net_pep_dim = args.max_len_pep * self.embedding_dim
#         self.net_tcr_dim = args.max_len_tcr * self.embedding_dim
#         self.net = nn.Sequential(
#             nn.Linear(self.net_pep_dim + self.net_tcr_dim,
#                       self.size_hidden1_dense),
#             nn.BatchNorm1d(self.size_hidden1_dense),
#             nn.Dropout(args.drop_rate * 2),
#             nn.SiLU(),
#             nn.Linear(self.size_hidden1_dense, self.size_hidden2_dense),
#             nn.BatchNorm1d(self.size_hidden2_dense),
#             nn.Dropout(args.drop_rate),
#             nn.SiLU(),
#             nn.Linear(self.size_hidden2_dense, 1),
#             nn.Sigmoid()
#         )

#     def forward(self, pep, tcr):

#         # Embedding
#         pep = self.embedding(pep) # batch * len * dim (25)
#         tcr = self.embedding(tcr) # batch * len * dim

#         pep = torch.transpose(pep, 0, 1)
#         tcr = torch.transpose(tcr, 0, 1)

#         # Attention
#         pep, pep_attn = self.attn_pep(pep,pep,pep)
#         tcr, tcr_attn = self.attn_tcr(tcr,tcr,tcr)

#         pep = torch.transpose(pep, 0, 1)
#         tcr = torch.transpose(tcr, 0, 1)

#         # Linear
#         pep = pep.reshape(-1, 1, pep.size(-2) * pep.size(-1))
#         tcr = tcr.reshape(-1, 1, tcr.size(-2) * tcr.size(-1))
#         peptcr = torch.cat((pep, tcr), -1).squeeze(-2)
#         peptcr = self.net(peptcr)

#         return peptcr

#modification-1
# import torch
# import torch.nn as nn
# import math

# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, max_len=5000):
#         super(PositionalEncoding, self).__init__()
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
#         # Handle odd d_model
#         pe[:, 0::2] = torch.sin(position * div_term[:(d_model + 1) // 2])  # Sine for even indices
#         pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])        # Cosine for odd indices
        
#         pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         x = x + self.pe[:, :x.size(1), :]
#         return x


# class Net(nn.Module):
#     def __init__(self, embedding, args):
#         super(Net, self).__init__()

#         # Embedding Layer
#         self.num_amino = len(embedding)
#         self.embedding_dim = len(embedding[0])
#         self.embedding = nn.Embedding(self.num_amino, self.embedding_dim, padding_idx=self.num_amino - 1)
#         if not (args.blosum is None or args.blosum.lower() == 'none'):
#             self.embedding = self.embedding.from_pretrained(torch.FloatTensor(embedding), freeze=False)

#         # Positional Encoding
#         self.positional_encoding = PositionalEncoding(self.embedding_dim, max_len=max(args.max_len_pep, args.max_len_tcr))

#         # Transformer Encoder
#         self.transformer_layer = nn.TransformerEncoderLayer(
#             d_model=self.embedding_dim,
#             nhead=args.heads,
#             dim_feedforward=args.lin_size,
#             dropout=args.drop_rate,
#             activation='relu'
#         )
#         self.transformer_encoder = nn.TransformerEncoder(self.transformer_layer, num_layers=2)

#         # Dense Layer
#         self.net = nn.Sequential(
#             nn.Linear(2 * self.embedding_dim, args.lin_size),
#             nn.BatchNorm1d(args.lin_size),
#             nn.Dropout(args.drop_rate),
#             nn.ReLU(),
#             nn.Linear(args.lin_size, 1),
#             nn.Sigmoid()
#         )

#     def forward(self, pep, tcr):
#         # Embedding and Positional Encoding
#         pep = self.embedding(pep)  # Shape: (batch, len, dim)
#         tcr = self.embedding(tcr)  # Shape: (batch, len, dim)
#         pep = self.positional_encoding(pep)
#         tcr = self.positional_encoding(tcr)

#         # Transformer Encoding
#         pep = self.transformer_encoder(pep.permute(1, 0, 2))  # Shape: (len, batch, dim)
#         tcr = self.transformer_encoder(tcr.permute(1, 0, 2))  # Shape: (len, batch, dim)

#         # Pooling
#         pep = torch.mean(pep, dim=0)  # Shape: (batch, dim)
#         tcr = torch.mean(tcr, dim=0)  # Shape: (batch, dim)

#         # Concatenate and Feed Forward
#         combined = torch.cat((pep, tcr), dim=-1)  # Shape: (batch, 2 * dim)
#         output = self.net(combined)

#         return output


#modificatoion-2
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Calculate div_term ensuring compatibility with odd d_model
        div_term = torch.exp(torch.arange(0, d_model // 2, dtype=torch.float) * -(torch.log(torch.tensor(10000.0)) / d_model))
        
        # Handle even and odd d_model separately
        self.encoding[:, 0:2 * div_term.size(0):2] = torch.sin(position * div_term)  # Even indices
        self.encoding[:, 1:2 * div_term.size(0):2] = torch.cos(position * div_term)  # Odd indices
        
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.encoding[:, :seq_len, :].to(x.device)

class Net(nn.Module):
    def __init__(self, embedding_matrix, args):
        super(Net, self).__init__()

        self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.positional_encoding = PositionalEncoding(d_model=self.embedding.embedding_dim, max_len=max(args.max_len_pep, args.max_len_tcr))

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.embedding.embedding_dim, nhead=args.heads, dropout=args.drop_rate)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        combined_dim = 2 * self.embedding.embedding_dim
        self.net = nn.Sequential(
            nn.Linear(combined_dim, args.lin_size),
            nn.ReLU(),
            nn.Dropout(args.drop_rate),
            nn.Linear(args.lin_size, 1),
            nn.Sigmoid()
        )

    def forward(self, pep, tcr):
        # Embed sequences
        pep = self.embedding(pep)
        tcr = self.embedding(tcr)

        # Add positional encoding
        pep = self.positional_encoding(pep)
        tcr = self.positional_encoding(tcr)

        # Pass through Transformer encoders
        pep = self.transformer_encoder(pep.permute(1, 0, 2))  # (seq_len, batch_size, embed_dim)
        tcr = self.transformer_encoder(tcr.permute(1, 0, 2))

        # Pooling (mean over sequence length)
        pep = torch.mean(pep, dim=0)  # (batch_size, embed_dim)
        tcr = torch.mean(tcr, dim=0)

        # Concatenate and pass through the feedforward network
        combined = torch.cat((pep, tcr), dim=-1)  # (batch_size, 2 * embed_dim)
        output = self.net(combined)
        return output
