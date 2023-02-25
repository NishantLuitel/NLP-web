import torch
import pickle
import math
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer

from torch.utils.data import dataset

from torch import Tensor, nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


def try_gpu(i=0):
    if torch.cuda.device_count() >= i+1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


device = try_gpu(0)

tokenizer = get_tokenizer(None)

with open('NLP_Trained_models/transformer_lm/transformer_vocab.pickle', 'rb') as f:
    vocab = pickle.load(f)


# Function for preprocessing input string
def data_process(raw_text_iter: dataset.IterableDataset) -> Tensor:
    """Converts raw text into a flat Tensor."""
    data = [torch.tensor(vocab(tokenizer(item)), dtype=torch.long)
            for item in raw_text_iter]
    return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))


def batchify(data: Tensor, bsz: int) -> Tensor:
    """Divides the data into bsz separate sequences, removing extra elements
    that wouldn't cleanly fit.
    Args:
        data: Tensor, shape [N]
        bsz: int, batch size
    Returns:
        Tensor of shape [N // bsz, bsz]
    """
    seq_len = data.size(0) // bsz
    data = data[:seq_len * bsz]
    data = data.view(bsz, seq_len).t().contiguous()
    return data.to(device)


def get_batch(source: Tensor, i: int) -> tuple[Tensor, Tensor]:
    """
    Args:
        source: Tensor, shape [full_seq_len, batch_size]
        i: int
    Returns:
        tuple (data, target), where data has shape [seq_len, batch_size] and
        target has shape [seq_len * batch_size]
    """
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].reshape(-1)
    #target = source[i+1:i+1+seq_len]
    return data, target


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2)
                             * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerModel(nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(
            d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, ntoken)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]
        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output


ntokens = len(vocab)  # size of vocabulary
emsize = 300  # embedding dimension
d_hid = 800  # dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 4  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 4  # number of heads in nn.MultiheadAttention
dropout = 0.05  # dropout probability
model = TransformerModel(ntokens, emsize, nhead, d_hid,
                         nlayers, dropout).to(device)
criterion = nn.CrossEntropyLoss()


bptt = 16


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)


model.load_state_dict(torch.load(
    'NLP_Trained_models/transformer_lm/best_model_3bigx3_corrected.pt'))
model.to(device)
best_model = model


softmax = nn.Softmax(dim=2)


def nonnaive_generator(model: nn.Module, gen_data: Tensor, no_words=5, k=50):
    model.eval()
    src_mask = generate_square_subsequent_mask(bptt).to(device)
    pred_text = []
    for i in range(no_words):
        # print('i:', i)
        batch_size = gen_data.size(0)
        if batch_size != bptt:
            src_mask_ = src_mask[:batch_size, :batch_size]
        output_softmax = model(gen_data, src_mask_)
        output_softmax_permuted = output_softmax.permute(1, 0, 2)
        indices = torch.topk(output_softmax_permuted, k, dim=2).indices.squeeze(0)

        values = torch.topk(softmax(output_softmax_permuted), k, dim=2).values
        values = values/torch.sum(values, dim=2, keepdims=True)
#         values = softmax(values)

#         values = torch.flip(values,dims = (2,))

        ind_sampled = torch.distributions.Categorical(values.squeeze(0)).sample()
        next_index = indices[-1][ind_sampled[-1]]

        # print('next word: ', vocab.lookup_token(next_index))
        #
        # print(i, "Values: ", values.squeeze(0)[-1], "Gen_data: ", gen_data,
        #       "possible tokens: ", indices[-1], "Pred_data: ", next_index)

        pred_text.append([vocab.lookup_token(next_index)][0])
        if(batch_size < 15):
            gen_data = torch.cat((gen_data[:, :], next_index.unsqueeze(0).unsqueeze(0)), 0)
            batch_size = gen_data.size(0)
        else:
            gen_data = torch.cat((gen_data[1:, :], next_index.unsqueeze(0).unsqueeze(0)), 0)
            batch_size = gen_data.size(0)

    return pred_text
