import torch
from torch import nn, Tensor

class DeepSpeech2(nn.Module):

    def __init__(
            self,
            n_feats,
            n_tokens,
            conv_channels=32,
            num_rnn_layers=3,
            rnn_hidden_dim=512,
            bidirectional_rnn=True,
    ):
        super().__init__()
        self.num_rnn_layers = num_rnn_layers

        self.conv_module = nn.Sequential(
            nn.Conv2d(
                1, 
                conv_channels,
                kernel_size=(41,11),
                stride=(2,2),
                padding=(20,5),
                bias=False,
            ),
            nn.BatchNorm2d(conv_channels),
            nn.SiLU(),
            nn.Conv2d(
                conv_channels,
                conv_channels,
                kernel_size=(21, 11),
                stride=(2, 1),
                padding=(10, 5),
                bias=False,
            ),
            nn.BatchNorm2d(conv_channels),
            nn.SiLU()
        )

        conv_output_shape = self._get_conv_output_shape(
            self.conv_module, fr_shape=n_feats
        )

        rnn_input_dim = conv_output_shape[1]*conv_output_shape[3]
        self.rnn_module = nn.ModuleList()
        rnn_output_dim = rnn_hidden_dim if not bidirectional_rnn else 2*rnn_hidden_dim
        for i in range(num_rnn_layers):
            self.rnn_module.append(
                nn.GRU(
                    input_size=rnn_input_dim if i == 0 else rnn_output_dim,
                    hidden_size=rnn_hidden_dim,
                    bidirectional=bidirectional_rnn,
                    dropout=0.1,
                    batch_first=True,
                )
            )
            self.rnn_module.append(nn.BatchNorm1d(rnn_output_dim))

            self.fc_head = nn.Sequential(
                nn.LayerNorm(rnn_output_dim),
                nn.Linear(rnn_output_dim, n_tokens),
            )

        def forward(self, spectrogram, spectrogram_length, **batch):

            log_spectrogram = torch.log(spectrogram + 1e-12)
            output = self.conv_module(
                log_spectrogram.transpose(1, 2).unsqueeze(1)
            )  # Now: (batch, layer, time, n_feats)
            batch_size, layers_cnt, seq_length, feats_dim = output.size()
            output = output.permute(0, 2, 1, 3).reshape(batch_size, seq_length, -1)

            for i in range(self.num_rnn_layers):
                output, _ = self.rnn_module[2 * i](output)
                output = self.rnn_module[2 * i + 1](output.permute(0, 2, 1)).permute(
                    0, 2, 1
                )  # BatchNorm1d accepts order (batch, n_feats, seq_len)

            output = self.fc_head(output)

            log_probs = nn.functional.log_softmax(output, dim=-1)
            log_probs_length = self.transform_input_lengths(spectrogram_length, seq_length)

            return {"log_probs": log_probs, "log_probs_length": log_probs_length}
        
        def transform_input_lengths(self, input_lengths, conv_output_length):
            output_lengths = torch.tensor(
                [conv_output_length] * input_lengths.size()[0]
            )  
            return output_lengths

        def __str__(self):
            """
            Model prints with the number of parameters.
            """
            all_parameters = sum([p.numel() for p in self.parameters()])
            trainable_parameters = sum(
                [p.numel() for p in self.parameters() if p.requires_grad]
            )

            result_info = super().__str__()
            result_info = result_info + f"\nAll parameters: {all_parameters}"
            result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

            return result_info
        
        @staticmethod
        def _get_conv_output_shape(conv_module, seq_len=100, fr_shape=100):
            inp = torch.tensor([1.0] * seq_len * fr_shape).view(
                1, 1, seq_len, fr_shape
            )  # Now: batch, layer, time, n_feats
            output = conv_module(inp)

            return output.shape