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
