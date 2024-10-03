import torch

class _ConvolutionModule(torch.nn.Module):

    def __init__(self, input_dim, num_channels, depthwise_kernel_size, dropout = 0.0) :
        super().__init__()
        if (depthwise_kernel_size - 1) % 2 != 0:
            raise ValueError("depthwise_kernel_size must be odd to achieve 'SAME' padding.")
        self.layer_norm = torch.nn.LayerNorm(input_dim)
        self.sequential = torch.nn.Sequential(
            torch.nn.Conv1d(input_dim, 2*num_channels,kernel_size= 1),
            torch.nn.GLU(dim=1),
            torch.nn.Conv1d(num_channels, num_channels, depthwise_kernel_size, padding=(depthwise_kernel_size-1) // 2, groups=num_channels),
            torch.nn.BatchNorm1d(num_channels),
            torch.nn.SiLU(),
            torch.nn.Conv1d(num_channels,input_dim, kernel_size=1),
            torch.nn.Dropout(dropout),
        )

    def forward(self, input: torch.Tensor) :
        x = self.layer_norm(input)
        x = x.transpose(1, 2)
        x = self.sequential(x)
        return x.transpose(1, 2)


class _FeedForwardModule(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim, dropout = 0.0) :
        super().__init__()
        self.sequential = torch.nn.Sequential(
            torch.nn.LayerNorm(input_dim),
            torch.nn.Linear(input_dim, hidden_dim, bias=True),
            torch.nn.SiLU(), torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, input_dim, bias=True),
            torch.nn.Dropout(dropout))

    def forward(self, input) :
        return self.sequential(input)


class ConformerLayer(torch.nn.Module):
    
    def __init__(self, input_dim: int, ffn_dim: int, num_attention_heads: int, depthwise_conv_kernel_size: int, dropout: float = 0.0) :
        super().__init__()

        self.ffn1 = _FeedForwardModule(input_dim, ffn_dim, dropout=dropout)
        self.self_attn_layer_norm = torch.nn.LayerNorm(input_dim)
        self.self_attn = torch.nn.MultiheadAttention(input_dim, num_attention_heads, dropout=dropout)
        self.self_attn_dropout = torch.nn.Dropout(dropout)
        self.conv_module = _ConvolutionModule(input_dim=input_dim, num_channels=input_dim, depthwise_kernel_size=depthwise_conv_kernel_size, dropout=dropout)
        self.ffn2 = _FeedForwardModule(input_dim, ffn_dim, dropout=dropout)
        self.final_layer_norm = torch.nn.LayerNorm(input_dim)

    def _apply_convolution(self, input) :
        residual = input
        input = input.transpose(0, 1)
        input = self.conv_module(input)
        input = input.transpose(0, 1)
        input = residual + input
        return input

    def forward(self, input) :

        residual = input
        x = self.ffn1(input)
        x = x * 0.5 + residual

        residual = x
        x = self.self_attn_layer_norm(x)
        x, _ = self.self_attn(query=x, key=x, value=x)
        x = self.self_attn_dropout(x)
        x = x + residual

        x = self._apply_convolution(x)

        residual = x
        x = self.ffn2(x)
        x = x * 0.5 + residual

        x = self.final_layer_norm(x)
        return x