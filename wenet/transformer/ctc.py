# Copyright (c) 2020 Mobvoi Inc. (authors: Binbin Zhang, Di Wu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified from ESPnet(https://github.com/espnet/espnet)

import torch
import torch.nn.functional as F
from typeguard import check_argument_types

from wenet.utils.common import remove_duplicates_and_blank
from wenet.utils.mask import make_pad_mask
from typing import Dict, Tuple

class CTC(torch.nn.Module):
    """CTC module"""
    def __init__(
        self,
        odim: int,
        encoder_output_size: int,
        dropout_rate: float = 0.0,
        reduce: bool = True,
        jpl: bool = False,
        jpl_conf: Dict = {},
    ):
        """ Construct CTC module
        Args:
            odim: dimension of outputs
            encoder_output_size: number of encoder projection units
            dropout_rate: dropout rate (0.0 ~ 1.0)
            reduce: reduce the CTC loss into a scalar
            jpl: wheter to use JPL(joint pseudo labelling) for unlabeled data 
            jpl_conf: Dict which includes parameters about jpl
                - ctc_coeff: loss coefficient for ctc
                - attention_coeff: loss coefficient for attention
                - transducer_coeff: loss coefficient for transducer
                - confidence_fn: type of confidence function for JPL
        """
        assert check_argument_types()
        super().__init__()
        eprojs = encoder_output_size
        self.dropout_rate = dropout_rate
        self.ctc_lo = torch.nn.Linear(eprojs, odim)

        self.jpl = jpl
        self.ctc_coeff: float = jpl_conf.get('ctc_coeff', 1.0)
        self.att_coeff: float = jpl_conf.get('attention_coeff', 1.0)
        self.rnnt_coeff: float = jpl_conf.get('transducer_coeff', 1.0)
        self.confidence_fn: str = jpl_conf.get('confidence_fn', 'one')
        self.label_fn: str = jpl_conf.get('label_fn', 'greedy')

        reduction_type = "sum" if reduce else "none"
        if self.jpl:
            reduction_type = "none"
        self.ctc_loss = torch.nn.CTCLoss(reduction=reduction_type)

    def ctc_pseudo_label(self, hlens: torch.Tensor, ys_hat: torch.Tensor, 
                         ys_pad: torch.Tensor, ys_lens: torch.Tensor):
        """ CTC-basd joint pseudo labelling (JPL)
        For every ys_pad with ys_len==0
        (which implies the text label is not provided for the waveform), 
        replace it pseudo-label made with ctc greedy search.
        """
        T, B, D = ys_hat.shape
        device = ys_hat.device
        loss_ctc_mask, loss_att_mask, loss_rnnt_mask = \
            [torch.ones(B, device=device) 
                for _ in range(3)]  # (B) each

        # 1. Select only unlabeled utterances in a batch
        index_mask = (ys_lens == 0)  # (B), True iff no text labels provided
        indices = torch.arange(B, device=device)  # (B)
        indices = torch.masked_select(indices, index_mask)  # (B'), where B' <= B
        B_ = indices.shape[0]
        if B_ == 0:
            return ys_pad, ys_lens, loss_ctc_mask, loss_att_mask, loss_rnnt_mask
        ys_hat_selected = ys_hat.transpose(0, 1)[indices]  # (B', T, D)

        # 2. Generate pseudo-labels by CTC greedy search
        if self.label_fn == 'greedy':
            topk_prob, topk_index = ys_hat_selected.topk(1, dim=2)  # (B', T, 1)
            topk_index = topk_index.view(B_, T)  # (B', T)
            mask = make_pad_mask(hlens[indices], T)  # (B', T)
            topk_index = topk_index.masked_fill_(mask, 0)  # (B', T)
            hyps: List[List[int]] = []
            for hyp in topk_index:
                hyp: List[int] = hyp.tolist()
                hyps.append(hyp)
            scores = topk_prob.sum(1).squeeze(1)  # (B') 
            hyps = [torch.tensor(remove_duplicates_and_blank(hyp), 
                                 device=device).type(ys_pad.type())
                    for hyp in hyps]  # list of variable length (l), where all l <= L
            hyps_lens = torch.tensor([len(hyp) for hyp in hyps], 
                                     device=device).type(ys_lens.type())  # (B')
            lmax = int(hyps_lens.max().item())  # int
            hyps_pad = torch.nn.utils.rnn.pad_sequence(hyps, batch_first=True)  # (B', lmax)
        else:
            assert False, "label_fn must be in {greedy}"
        # 3. Merge labels and pseudo-labels
        Lmax = int(ys_lens.max().item())  # int
        print('B=%d, B_=%d, Lmax=%d, lmax=%d, T=%d' % (B, B_, Lmax, lmax, T))
        print('hyps_lens', hyps_lens, 'hyps_pad', hyps_pad)
        if lmax > Lmax:  # lmax > Lmax
            ys_pad = torch.nn.functional.pad(ys_pad, (0, lmax - Lmax))  # (B, lmax)
        new_ys_pad = ys_pad.clone()
        new_ys_pad[indices, :lmax] = hyps_pad  # (B', lmax)
        new_ys_lens = ys_lens.clone()  # (B')
        new_ys_lens[indices] = hyps_lens  # (B')

        # 4. Loss coefficients
        if self.confidence_fn == 'one':
            confidence = torch.ones(B_, device=device)  # (B')
        elif self.confidence_fn == 'prob':
            confidence = torch.exp(scores.detach())  # (B')
        else:
            assert False, "confidence_fn should be in {ones}"
        loss_ctc_mask[indices] *= confidence * self.ctc_coeff  # (B')
        loss_att_mask[indices] *= confidence * self.att_coeff  # (B')
        loss_rnnt_mask[indices] *= confidence * self.rnnt_coeff  # (B')
        return new_ys_pad, new_ys_lens, loss_ctc_mask, loss_att_mask, loss_rnnt_mask

    def forward(
        self, hs_pad: torch.Tensor, hlens: torch.Tensor,
        ys_pad: torch.Tensor, ys_lens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculate CTC loss.

        Args:
            hs_pad: batch of padded hidden state sequences (B, Tmax, D)
            hlens: batch of lengths of hidden state sequences (B)
            ys_pad: batch of padded character id sequence tensor (B, Lmax)
            ys_lens: batch of lengths of character sequence (B)
        """
        # hs_pad: (B, T, NProj) -> ys_hat: (B, T, Nvocab)
        ys_hat = self.ctc_lo(F.dropout(hs_pad, p=self.dropout_rate))
        # ys_hat: (B, T, D) -> (T, B, D)
        ys_hat = ys_hat.transpose(0, 1)
        ys_hat = ys_hat.log_softmax(2)

        if self.training and self.jpl:
            new_ys_pad, new_ys_lens, loss_ctc_mask, loss_att_mask, loss_rnnt_mask \
                = self.ctc_pseudo_label(hlens, ys_hat, ys_pad, ys_lens)
        else:
            new_ys_pad = ys_pad
            new_ys_lens = ys_lens
            # fake mask tensors
            loss_ctc_mask, loss_att_mask, loss_rnnt_mask = \
                [torch.ones(0, device=ys_hat.device) for _ in range(3)] 

        loss = self.ctc_loss(ys_hat, ys_pad, hlens, ys_lens)  # scalar
        if self.jpl:
            # In this case, loss shape is (B) rather than scalar
            if self.training:
                loss = loss * loss_ctc_mask  # (B)
            loss = loss.sum()  # scalar

        # Batch-size average
        loss = loss / ys_hat.size(1)
        # NOTE(hslee): not to return ys_pad and ys_lens, since they are 
        #              already modified by inplace operations
        return loss, loss_att_mask, loss_rnnt_mask, new_ys_pad, new_ys_lens 

    def log_softmax(self, hs_pad: torch.Tensor) -> torch.Tensor:
        """log_softmax of frame activations

        Args:
            Tensor hs_pad: 3d tensor (B, Tmax, eprojs)
        Returns:
            torch.Tensor: log softmax applied 3d tensor (B, Tmax, odim)
        """
        return F.log_softmax(self.ctc_lo(hs_pad), dim=2)

    def argmax(self, hs_pad: torch.Tensor) -> torch.Tensor:
        """argmax of frame activations

        Args:
            torch.Tensor hs_pad: 3d tensor (B, Tmax, eprojs)
        Returns:
            torch.Tensor: argmax applied 2d tensor (B, Tmax)
        """
        return torch.argmax(self.ctc_lo(hs_pad), dim=2)
