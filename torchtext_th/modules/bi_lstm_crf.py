from typing import Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from torchtext_th.modules.crf import ConditionalRandomField


class BiLSTMCRF(nn.Module):

    def __init__(self, **kwargs: Dict[str, Any]):
        """
        BiLSTM with CRF sequence labeling model

        Parameters
        ----------
        kwargs:
            model configuration with these required fields:
                - vocab_count: int
                - label_count: int
                - embedding_dim: int
                - lstm_hidden_dim: int
                - lstm_num_layers: int
                - constraints: Optional[List[Tuple[int]]] = None
                - p_drop: float = 0.5
        """
        super().__init__()

        self.embedding_dim: int = kwargs.get("embedding_dim")
        self.lstm_hidden_dim: int = kwargs.get("lstm_hidden_dim")
        self.p_drop: float = kwargs.get("p_drop", 0.5)

        vocab_count: int = kwargs.get("vocab_count")
        label_count: int = kwargs.get("label_count")
        lstm_hidden_dim: int = kwargs.get("lstm_hidden_dim")
        lstm_num_layers: int = kwargs.get("lstm_num_layers")
        constraints: int = kwargs.get("constraints", None)

        ######################
        # 1. Embedding layer #
        ######################
        self.embedding = self._init_embedding(vocab_count, self.embedding_dim)

        #######################
        # 2. Sequence encoder #
        #######################
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm = nn.LSTM(input_size=self.embedding_dim,
                            hidden_size=lstm_hidden_dim // 2,
                            num_layers=lstm_num_layers,
                            bidirectional=True,
                            batch_first=True)

        #####################
        # 3. Label emission #
        #####################
        self.label_count = label_count
        self.dense = nn.Linear(in_features=lstm_hidden_dim,
                               out_features=self.label_count)
        xavier_uniform_(self.dense.weight)

        ####################
        # 4. Label decoder #
        ####################
        self.crf = ConditionalRandomField(num_tags=self.label_count,
                                          constraints=constraints,
                                          include_start_end_transitions=True)

    def _init_embedding(self, num_embeddings: int, embedding_dim: int,
                        padding_idx: int = 0) -> nn.Embedding:
        emb_layer = nn.Embedding(num_embeddings=num_embeddings,
                                 embedding_dim=embedding_dim,
                                 padding_idx=padding_idx)
        xavier_uniform_(emb_layer.weight)
        return emb_layer

    def loss(self, pred_dict: Dict[str, torch.Tensor],
             target: torch.Tensor) -> torch.Tensor:
        emission_logit: torch.Tensor = pred_dict.get("emission_logit")
        mask: torch.Tensor = pred_dict.get("mask")
        return self.crf.neg_log_likelihood(emission_logits=emission_logit,
                                           tags=target,
                                           mask=mask)

    @staticmethod
    def get_seq_mask(lengths: torch.Tensor):
        batch_size = lengths.numel()
        max_len = lengths.max()
        return (torch.arange(0, max_len)
                .type_as(lengths)
                .repeat(batch_size, 1)
                .lt(lengths.unsqueeze(1)))

    def _apply_lstm(self, x: torch.Tensor, seq_len: int,
                    x_length: torch.Tensor):
        # lstm_out: (batch_size, seq_len, lstm_hidden_dim)
        #    - lstm_out will be zero vectors at the padding position
        #    - Check this link for the detail about pack_padded_sequence
        #        - https://gist.github.com/HarshTrivedi/f4e7293e941b17d19058f6fb90ab0fec
        packed_emb_x = pack_padded_sequence(x, x_length,
                                            batch_first=True)
        packed_lstm_out, _ = self.lstm(packed_emb_x)
        lstm_out, _ = pad_packed_sequence(packed_lstm_out, batch_first=True,
                                          total_length=seq_len)
        return lstm_out

    def _projection_to_label(self, x: torch.Tensor, batch_size: int,
                             seq_len: int):

        # flatten_x: (batch_size * seq_len, lstm_hidden_dim)
        flatten_x = x.contiguous().view(-1, self.lstm_hidden_dim)

        # flatten_emission_logit: (batch_size * seq_len, label_count)
        flatten_emission_logit = self.dense(flatten_x)

        # emission_logit: (batch_size, seq_len, label_count)
        emission_logit = flatten_emission_logit.view(batch_size, seq_len, -1)

        return emission_logit

    def forward(self,
                input_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # x: (batch_size, seq_len)
        x: torch.Tensor = input_dict.get("x")
        x_length: torch.Tensor = input_dict.get("x_length")
        mask: torch.Tensor = input_dict.get("mask", None)

        batch_size, seq_len = x.shape

        # print("Forward", batch_size, seq_len)
        # print(x_length)

        ################
        # 1. Embedding #
        ################
        # embedded_x: (batch_size, seq_len, embedding_dim)
        embedded_x = self.embedding(x)
        embedded_x = F.dropout(embedded_x,
                               p=self.p_drop,
                               training=self.training)

        ###########
        # 2. LSTM #
        ###########
        lstm_out = self._apply_lstm(
            x=embedded_x,
            seq_len=seq_len,
            x_length=x_length)
        lstm_out = F.dropout(lstm_out, p=self.p_drop, training=self.training)

        ############
        # 3. Dense #
        ############
        emission_logit = self._projection_to_label(
            x=lstm_out,
            batch_size=batch_size,
            seq_len=seq_len
        )

        ##################
        # 4. CRF decoder #
        ##################
        if mask is None:
            mask = __class__.get_seq_mask(x_length)
        best_paths = self.crf(emission_logit, mask)

        # Just get the tags and ignore the score.
        predicted_tags = [tags for tags, score in best_paths]

        output_dict = dict(
            emission_logit=emission_logit,
            predicted_tags=predicted_tags
        )

        return output_dict


class CNNBiLSTMCRF(BiLSTMCRF):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.p_drop_cnn = kwargs.get("p_drop_cnn", 0.2)
        cnn_filter_sizes = kwargs.get("cnn_filter_sizes", [2, 3, 4])
        cnn_filter_nums = kwargs.get("cnn_filter_nums", None)

        # keep the original dimension after CNN if cnn_filter_nums is None
        if cnn_filter_nums is None:
            cnn_filter_count = len(cnn_filter_sizes)
            divided_size = self.embedding_dim // cnn_filter_count
            cnn_filter_nums = [divided_size] * cnn_filter_count
            cnn_filter_nums[0] += self.embedding_dim - sum(cnn_filter_nums)

        self.conv_layers = self._init_convolution(
            filter_sizes=cnn_filter_sizes,
            in_channel=self.embedding_dim,
            out_channels=cnn_filter_nums
        )

    def _init_convolution(self, filter_sizes: List[int],
                          in_channel: int, out_channels: List[int]):
        conv_layers = nn.ModuleList()
        for i in range(len(filter_sizes)):
            filter_size = filter_sizes[i]
            out_channel = out_channels[i]
            if filter_size % 2 == 0:
                msg = f"Filter size should be an odd number " \
                      f"(given {filter_size})"
                raise ValueError(msg)

            # to keep the original sequence length
            padding_size = (filter_size - 1) // 2
            conv = nn.Conv1d(in_channels=in_channel,
                             out_channels=out_channel,
                             kernel_size=filter_size,
                             padding=padding_size,
                             stride=1)
            xavier_uniform_(conv.weight)
            conv_layers.append(conv)
        return conv_layers

    def _apply_cnn(self, x: torch.Tensor):
        """
        Apply CNN filters to input sequence

        Parameters
        ----------
        x: (batch_size, seq_len, input_feature)

        Returns
        -------
        (batch_size, seq_len, output_feature)
        """
        # x_permuted: (batch_size, input_feature, seq_len)
        x_permuted = x.permute(0, 2, 1)
        conv_out = []
        for conv in self.conv_layers:
            conv_out_i = F.relu(conv(x_permuted))
            conv_out_i = F.dropout(conv_out_i, p=self.p_drop_cnn,
                                   training=self.training)
            conv_out.append(conv_out_i)

        # output: (batch_size, seq_len, output_feature)
        output = torch.cat(conv_out, dim=-1).permute(0, 2, 1)
        return output

    def forward(self,
                input_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # x: (batch_size, seq_len)
        x: torch.Tensor = input_dict.get("x")
        x_length: torch.Tensor = input_dict.get("x_length")
        mask: torch.Tensor = input_dict.get("mask", None)

        batch_size, seq_len = x.shape

        ################
        # 1. Embedding #
        ################
        # embedded_x: (batch_size, seq_len, embedding_dim)
        embedded_x = self.embedding(x)
        embedded_x = F.dropout(embedded_x,
                               p=self.p_drop,
                               training=self.training)
        ##########
        # 2. CNN #
        ##########
        # cnn_out: (batch_size, seq_len, cnn_features)
        cnn_out = self._apply_cnn(embedded_x)

        ###########
        # 3. LSTM #
        ###########
        lstm_out = self._apply_lstm(x=cnn_out,
                                    seq_len=seq_len,
                                    x_length=x_length)
        lstm_out = F.dropout(lstm_out, p=self.p_drop, training=self.training)

        ############
        # 4. Dense #
        ############
        emission_logit = self._projection_to_label(
            x=lstm_out,
            batch_size=batch_size,
            seq_len=seq_len
        )

        ##################
        # 5. CRF decoder #
        ##################
        if mask is None:
            mask = __class__.get_seq_mask(x_length)
        best_paths = self.crf(emission_logit, mask)

        # Just get the tags and ignore the score.
        predicted_tags = [tags for tags, score in best_paths]

        output_dict = dict(
            emission_logit=emission_logit,
            predicted_tags=predicted_tags
        )

        return output_dict

