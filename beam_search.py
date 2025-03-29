import torch
import torch.nn as nn


class BeamSearchTransformerTransducer:
    r"""
    Transformer Transducer Beam Search
    Args:
        joint: Joint network to combine encoder and decoder outputs.
        decoder: Decoder (TransformerTransducerDecoder).
        beam_size (int): Size of the beam.
        expand_beam (float): Threshold to limit expanded hypotheses.
        state_beam (float): Threshold to compare hypotheses.
        blank_id (int): ID of the blank token.
    """
    def __init__(
        self,
        joint,
        decoder,
        beam_size: int = 3,
        expand_beam: float = 2.3,
        state_beam: float = 4.6,
        blank_id: int = 3,
        sos_id: int = 1
    ) -> None:
        self.joint = joint
        self.decoder = decoder
        self.forward_step = self.decoder.forward_step
        self.beam_size = beam_size
        self.expand_beam = expand_beam
        self.state_beam = state_beam
        self.blank_id = blank_id
        self.sos_id = sos_id

    def _fill_sequence(self, hypotheses):
        max_len = max([len(hyp) for hyp in hypotheses])
        batch = len(hypotheses)
        padded = torch.zeros((batch, max_len), dtype=torch.long)

        for i, hyp in enumerate(hypotheses):
            padded[i, :len(hyp)] = hyp

        return padded

    def forward(self, encoder_outputs: torch.Tensor, max_length: int):
        """
        Beam search decoding.
        Args:
            encoder_outputs (torch.FloatTensor): Encoder outputs of shape (batch, seq_length, dimension).
            max_length (int): Maximum decoding length.

        Returns:
            torch.LongTensor: Model predictions.
        """
        hypothesis = list()
        hypothesis_score = list()

        for batch_idx in range(encoder_outputs.size(0)):
            blank = torch.ones((1, 1), device=encoder_outputs.device, dtype=torch.long) * self.blank_id
            step_input = torch.ones((1, 1), device=encoder_outputs.device, dtype=torch.long) * self.sos_id
            hyp = {
                "prediction": [self.sos_id],
                "logp_score": 0.0,
            }
            ongoing_beams = [hyp]

            for t_step in range(max_length):
                process_hyps = ongoing_beams
                ongoing_beams = list()

                while True:
                    if len(ongoing_beams) >= self.beam_size:
                        break

                    a_best_hyp = max(process_hyps, key=lambda x: x["logp_score"] / len(x["prediction"]))

                    if len(ongoing_beams) > 0:
                        b_best_hyp = max(
                            ongoing_beams,
                            key=lambda x: x["logp_score"] / len(x["prediction"]),
                        )

                        a_best_prob = a_best_hyp["logp_score"]
                        b_best_prob = b_best_hyp["logp_score"]

                        if b_best_prob >= self.state_beam + a_best_prob:
                            break

                    process_hyps.remove(a_best_hyp)

                    step_input[0, 0] = a_best_hyp["prediction"][-1]
                    step_lengths = encoder_outputs.new_tensor([0], dtype=torch.long)

                    step_outputs = self.forward_step(step_input, step_lengths).squeeze(0).squeeze(0)
                    log_probs = self.joint(encoder_outputs[batch_idx, t_step, :], step_outputs)

                    topk_targets, topk_idx = log_probs.topk(k=self.beam_size)

                    if topk_idx[0] != blank:
                        best_logp = topk_targets[0]
                    else:
                        best_logp = topk_targets[1]

                    for j in range(topk_targets.size(0)):
                        topk_hyp = {
                            "prediction": a_best_hyp["prediction"][:],
                            "logp_score": a_best_hyp["logp_score"] + topk_targets[j],
                        }

                        if topk_idx[j] == self.blank_id:
                            ongoing_beams.append(topk_hyp)
                            continue

                        if topk_targets[j] >= best_logp - self.expand_beam:
                            topk_hyp["prediction"].append(topk_idx[j].item())
                            process_hyps.append(topk_hyp)

            ongoing_beams = sorted(
                ongoing_beams,
                key=lambda x: x["logp_score"] / len(x["prediction"]),
                reverse=True,
            )[0]

            hypothesis.append(torch.LongTensor(ongoing_beams["prediction"][1:]))
            hypothesis_score.append(ongoing_beams["logp_score"] / len(ongoing_beams["prediction"]))

        return self._fill_sequence(hypothesis)
