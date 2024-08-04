import torch
import torch.nn as nn

# Classes are arranged by levels of abstraction
# from bottom to top


class SelfAttention (nn.Module):
    def __init__(self, embed_size, heads):
        """Initialize SelfAttention .

        Args:
            embed_size (int-like): size to make the embed vector
            heads (int): number of attention heads
        """
        super(SelfAttention, self).__init__()

        self.embed_size = embed_size
        self.heads = heads

        # embed_size // heads for narrow; embed_size for wide
        self.head_dim = embed_size // heads

        assert (self.head_dim * heads ==
                embed_size), "embed size needs to be divisible by heads"

        # adding paramaters for values keys and queries
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)

        # heads * head_dim will always equal embed_size
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        """forward computation of attention

        Args:
            values (tensor(N, len, embed_size)): values matrix for self attention
            keys (tensor(N, len, embed_size)): keys matrix for self attention
            query (tensor(N, len, embed_size)): queries matrix for self attention
            mask (mask): matrix of 1s and 0s where 0s denote a value to be skipped
                         during self attention computation

        Returns:
            [tensor(N, query_len , embed_size)]: output matrix after self attention
        """
        N = query.shape[0]

        # print(N)
        # print(
        #     f'Values: {values.shape} , Keys: {keys.shape} , Query: {query.shape}')

        # value, key and query len will always be == seq_len
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # equivelent to b, t, h, k from Basic Self Attention
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        query = query.reshape(N, query_len, self.heads, self.head_dim)

        # adds paramaters for the model to train for K, Q, V
        # doesnt change shapes at all
        values = self.values(values)
        keys = self.keys(keys)
        query = self.queries(query)

        # queries.shape (N , query_len , heads , head_dim)
        # values and keys will be the same with their respective lengths
        # energy.shape (N , heads , query_len , key_len)
        # transpose before bmm
        energy = torch.einsum('nqhd,nkhd->nhqk', [query, keys])
        # einsum totally replaces bmm and flatten

        if mask is not None:
            # sets all values in energy to -infinity
            # if they are set as 0 in the mask
            energy = energy.masked_fill_(mask == 0, float('-1e20'))

        # softmaxes the energy aka w' making it into attention aka weights
        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=3)

        # attention.shape = (N , heads , query_len , key_len)
        # values.shape = (N , value_len , heads , head_dim)
        # out.shape (N , query_len , heads, head_dim)
        out = (
            torch.einsum(
                "nhql,nlhd->nqhd",
                [attention, values])
            .reshape(N, query_len, self.heads * self.head_dim)
        )  # then concat heads with heads dim

        out = self.fc_out(out)

        return out


class TransformerBlock (nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()

        self.attention = SelfAttention(embed_size, heads)

        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.ff = nn.Sequential(
            # Runs input through a series of linear layers
            # Input size is embed_size as is output size
            # During linear layers the input is transformed by the product of
            # embed size and forward expansion then transformed back to embed size
            nn.Linear(embed_size, forward_expansion*embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)

        # applies dropout and layernorm to attention
        # attention + query is the skip connection similar
        # to the skip connections from resnet
        x = self.dropout(self.norm1(attention + query))

        # Runs x through the feedforward layers
        forward = self.ff(x)

        # Once again adds normalization and dropout
        # Skip connection is also included
        out = self.dropout(self.norm2(forward + x))

        return out


class Encoder (nn.Module):
    def __init__(
            self,
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_len
    ):

        super(Encoder, self).__init__()

        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.pos_embedding = nn.Embedding(max_len, embed_size)

        self.layers = nn.ModuleList(modules=[
            TransformerBlock(
                embed_size,
                heads,
                dropout,
                forward_expansion
            )

            for _ in range(num_layers)
        ]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self,
                x,
                mask):

        N, seq_len = x.shape

        # create scalar positions
        # these are used to allow the model to understand sequence order
        positions = torch.arange(0, seq_len).expand(N, seq_len).to(self.device)

        # Create position and word embeddings then merge them together
        # Dropout is applied afterward
        out = self.dropout(
            self.word_embedding(x) +
            self.pos_embedding(positions)
        )

        # Run
        for layer in self.layers:
            out = layer(out, out, out, mask)

        return out


class DecoderBlock (nn.Module):
    def __init__(
            self,
            embed_size,
            heads,
            forward_expansion,
            dropout,
            device
    ):

        super(DecoderBlock, self).__init__()

        self.attention = SelfAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)

        self.transformer_block = TransformerBlock(
            embed_size,
            heads,
            dropout,
            forward_expansion
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self,
                x,
                value,
                key,
                src_mask,
                trg_mask,
                ):
        # src mask is optional it deals with padding

        # Computes masked self attention
        attention = self.attention(x, x, x, trg_mask)

        # Runs dropout and layer norm on attention
        # adding x creates the resiudal skip connection
        query = self.dropout(self.norm(attention + x))

        # Normal transformer block
        # Just unmasked attention
        out = self.transformer_block(value,
                                     key,
                                     query,
                                     src_mask)

        return out


class Decoder (nn.Module):
    def __init__(
            self,
            trg_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            device,
            max_len
    ):

        super(Decoder, self).__init__()

        self.device = device

        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.pos_embedding = nn.Embedding(max_len, embed_size)

        self.layers = nn.ModuleList(
            [
                DecoderBlock(
                    embed_size,
                    heads,
                    forward_expansion,
                    dropout,
                    device
                )
                for _ in range(num_layers)
            ]
        )

        # Outputs a probability vector for each word
        self.fc_out = nn.Linear(embed_size, trg_vocab_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self,
                x,
                enc_out,
                src_mask,
                trg_mask):
        N, seq_len = x.shape

        positions = (torch.arange(0, seq_len)
                     .expand(N, seq_len)
                     .to(self.device)
                     )

        # creates and merges word and positional embedding vectors
        # this allows the model to understand sequence order
        x = self.dropout(self.word_embedding(x) +
                         self.pos_embedding(positions))

        for layer in self.layers:
            # Feeds current predictions or sos token
            # into the decoder block for self attention
            # and transformation with linear layers
            # Encoder output is also added to help the model
            # understand the input sequence
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)

        out = self.fc_out(x)

        return out


class Transformer (nn.Module):
    def __init__(
            self,
            src_vocab_size,
            trg_vocab_size,
            src_pad_idx,
            trg_pad_idx,
            embed_size,
            num_layers,
            forward_expansion,
            heads,
            dropout,
            device,
            max_len
    ):
        """Initialize Transformer. Creates src and target masks.
           It also houses the encoder and decoder. Calling it
           will run the forward

        Args:
            src_vocab_size (int): size of the input vocabulary
            trg_vocab_size (int): size of the target vocab
            src_pad_idx (int/str): padding token or number
            trg_pad_idx (int/str): padding token or number
            embed_size (int): size of the embedding vecotor to create
            num_layers (int): number of layers to apply to encoder and decoder
            forward_expansion ([type]): amount to upscale linear layers inside of
                                        transformer. They will be downsampled so it does
                                        not affect other hyperparamters
            heads (int): attention heads for self attention
            dropout (decimal_int): dropout rate for all dropout layers
            device (torch.device/str): device to run computations
            max_len (int): maximum sentence length for both src and trg
        """
        super(Transformer, self).__init__()

        self.name = "Basic_Transformer"

        self.encoder = Encoder(
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_len
        )

        self.decoder = Decoder(
            trg_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            device,
            max_len
        )

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        # trg pad idx isnt used at all

        self.device = device

    def make_src_mask(self, src):
        # src_mask.shape = (N , 1 , 1 , src_len)
        # if a part of the input (src) is a padding token
        # it is set to zero in the mask, otherwise it is set to one
        # this is used to remove needless computation on padding tokens
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

        return src_mask.to(self.device)

    def make_trg_mask(self, trg):
        # get batch size and trg_len from target shape
        N, trg_len = trg.shape

        # creates the target mask to stop the model
        # from seeing future values in its computation
        # the mask is 0s for all values that are masked out
        # it is applied to the energy or w' before softmax
        # basically it removes all computation for values
        # that would be in the future for the models sentence
        trg_mask = (
            # torch.tril makes a lower triangular mask
            # by setting all values in the upper right corner to 0
            torch.tril(
                torch.ones(
                    (trg_len, trg_len))
            )
            # create a mask for each
            .expand(N, 1, trg_len, trg_len))

        return trg_mask.to(self.device)

    def forward(self, src, trg):
        """Compute the forward pass through the layer .

        Args:
            src (tensor (N , src_len)): input sentence for encoder
            trg (tensor (N , trg_len)): target sentence

        Returns:
            tensor (N, seq_len, trg_vocab_size): output from the decoder
        """

        # creates masks
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)

        # runs encoder step
        enc_src = self.encoder(
            src, src_mask
        )

        # runs decoder step
        dec_out = self.decoder(
            trg,
            enc_src,
            src_mask,
            trg_mask
        )

        return dec_out


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(
        device
    )
    trg = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], [
                       1, 5, 6, 2, 4, 7, 6, 2]]).to(device)

    print(f'Src shape: {x.shape} , Trg shape: {trg.shape}')

    src_pad_idx = 0
    trg_pad_idx = 0
    src_vocab_size = 10
    trg_vocab_size = 10

    embed_size = 512
    num_layers = 6
    forward_expansion = 4
    heads = 8
    dropout = 0
    max_len = 100

    model = Transformer(
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        trg_pad_idx,
        embed_size,
        num_layers,
        forward_expansion,
        heads,
        dropout,
        device,
        max_len).to(device)

    out = model(x, trg[:, :-1])
    print(out.shape)