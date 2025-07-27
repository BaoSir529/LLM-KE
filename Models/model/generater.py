r"""undocumented"""

import torch
from torch import nn
from fastNLP.models.seq2seq_model import Seq2SeqModel
from fastNLP.modules.decoder.seq2seq_decoder import Seq2SeqDecoder, State
import torch.nn.functional as F
from fastNLP.core.utils import _get_model_device
from functools import partial


class SequenceGeneratorModel(nn.Module):
    def __init__(self,
                 seq2seq_model: Seq2SeqModel,
                 bos_token_id,
                 eos_token_id=None,
                 max_length=30,
                 max_len_a=0.0,
                 num_beams=1,
                 do_sample=True,
                 sc_only=False,
                 repetition_penalty=1,
                 length_penalty=1.0,
                 pad_token_id=0,
                 restricter=None):
        super().__init__()
        self.seq2seq_model = seq2seq_model
        self.restricter = restricter
        self.sc_only = sc_only
        self.generator = SequenceGenerator(seq2seq_model.decoder,
                                           max_length=max_length,
                                           max_len_a=max_len_a,
                                           num_beams=num_beams,
                                           do_sample=do_sample,
                                           sc_only=sc_only,
                                           bos_token_id=bos_token_id,
                                           eos_token_id=eos_token_id,
                                           repetition_penalty=repetition_penalty,
                                           length_penalty=length_penalty,
                                           pad_token_id=pad_token_id,
                                           restricter=restricter)

    def forward(self,
                input_ids,
                image_features,
                attention_mask=None,
                aesc_infos=None,
                sentence_length=None,
                first=None):
        """
        透传调用seq2seq_model的forward
        :param torch.LongTensor src_tokens: bsz x max_len
        :param torch.LongTensor tgt_tokens: bsz x max_len'
        :param torch.LongTensor src_seq_len: bsz
        :param torch.LongTensor tgt_seq_len: bsz
        :return:
        """
        return self.seq2seq_model(input_ids=input_ids,
                                  image_features=image_features,
                                  attention_mask=attention_mask,
                                  sentence_length=sentence_length,
                                  aesc_infos=aesc_infos)

    def predict(self,
                input_ids,
                image_features,
                attention_mask=None,
                aesc_infos=None,
                sentence_length=None):
        """
        给定source的内容，输出generate的内容
        :param torch.LongTensor src_tokens: bsz x max_len
        :param torch.LongTensor src_seq_len: bsz
        :return:
        """
        state = self.seq2seq_model.prepare_state(input_ids, image_features, attention_mask, length=sentence_length)
        tgt_tokens = aesc_infos['labels'].to(input_ids.device)
        # print()
        if self.sc_only:
            result = self.generator.generate(state,
                                             tokens=tgt_tokens[:, :3],
                                             gt_tokens=tgt_tokens)
        else:
            result = self.generator.generate(state, tokens=tgt_tokens[:, :3])  # the prompt is provided to the model
        return result

__all__ = ['SequenceGenerator']


class SequenceGenerator:
    def __init__(self,
                 decoder: Seq2SeqDecoder,
                 max_length=20,
                 max_len_a=0.0,
                 num_beams=1,
                 do_sample=False,
                 sc_only=False,
                 bos_token_id=None,
                 eos_token_id=None,
                 repetition_penalty=1,
                 length_penalty=1.0,
                 pad_token_id=0,
                 restricter=None):

        self.generate_func = partial(greedy_generate,
                                     decoder=decoder,
                                     max_length=max_length,
                                     max_len_a=max_len_a,
                                     num_beams=num_beams,
                                     sc_only=sc_only,
                                     bos_token_id=bos_token_id,
                                     eos_token_id=eos_token_id,
                                     repetition_penalty=repetition_penalty,
                                     length_penalty=length_penalty,
                                     pad_token_id=pad_token_id,
                                     restricter=restricter)
        self.do_sample = do_sample
        self.max_length = max_length
        self.num_beams = num_beams
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.repetition_penalty = repetition_penalty
        self.length_penalty = length_penalty
        self.decoder = decoder
        self.pad_token_id = pad_token_id
        self.restricter = restricter
        self.max_len_a = max_len_a

    def set_new_generator(self,
                          max_length=-1,
                          max_len_a=-1,
                          num_beams=-1,
                          repetition_penalty=-1,
                          length_penalty=-1,
                          restricter=-1):
        if max_length == -1:
            max_length = self.max_length
        if max_len_a == -1:
            max_len_a = self.max_len_a
        if num_beams == -1:
            num_beams = self.num_beams
        if repetition_penalty == -1:
            repetition_penalty = self.repetition_penalty
        if length_penalty == -1:
            length_penalty = self.length_penalty
        if restricter == -1:
            restricter = self.restricter
        self.generate_func = partial(greedy_generate,
                                     decoder=self.decoder,
                                     max_length=max_length,
                                     max_len_a=max_len_a,
                                     num_beams=num_beams,
                                     sc_only=sc_only,
                                     bos_token_id=self.bos_token_id,
                                     eos_token_id=self.eos_token_id,
                                     repetition_penalty=repetition_penalty,
                                     length_penalty=length_penalty,
                                     pad_token_id=self.pad_token_id,
                                     restricter=restricter)

    @torch.no_grad()
    def generate(self, state, tokens=None, gt_tokens=None):
        return self.generate_func(tokens=tokens,
                                  gt_tokens=gt_tokens,
                                  state=state)


@torch.no_grad()
def greedy_generate(decoder,
                    tokens=None,
                    gt_tokens=None,
                    state=None,
                    sc_eval=False,
                    max_length=20,
                    max_len_a=0.0,
                    num_beams=1,
                    sc_only=False,
                    bos_token_id=None,
                    eos_token_id=None,
                    pad_token_id=0,
                    repetition_penalty=1,
                    length_penalty=1.0,
                    restricter=None):

    if sc_only:
        token_ids = sc_generate(decoder,
                                tokens=tokens,
                                gt_tokens=gt_tokens,
                                state=state,
                                max_length=max_length,
                                max_len_a=max_len_a,
                                bos_token_id=bos_token_id,
                                eos_token_id=eos_token_id,
                                repetition_penalty=repetition_penalty,
                                length_penalty=length_penalty,
                                pad_token_id=pad_token_id,
                                restricter=restricter)
        return token_ids
    if num_beams == 1:
        token_ids = _no_beam_search_generate(decoder,
                                            tokens=tokens,
                                            state=state,
                                            max_length=max_length,
                                            max_len_a=max_len_a,
                                            bos_token_id=bos_token_id,
                                            eos_token_id=eos_token_id,
                                            repetition_penalty=repetition_penalty,
                                            length_penalty=length_penalty,
                                            pad_token_id=pad_token_id,
                                            restricter=restricter)
    else:
        token_ids = _beam_search_generate(decoder,
                                            tokens=tokens,
                                            state=state,
                                            max_length=max_length,
                                            max_len_a=max_len_a,
                                            num_beams=num_beams,
                                            bos_token_id=bos_token_id,
                                            eos_token_id=eos_token_id,
                                            do_sample=False,
                                            repetition_penalty=repetition_penalty,
                                            length_penalty=length_penalty,
                                            pad_token_id=pad_token_id,
                                            restricter=restricter)

    return token_ids


def _no_beam_search_generate(decoder: Seq2SeqDecoder,
                             state,
                             tokens=None,
                             max_length=20,
                             max_len_a=0.0,
                             bos_token_id=None,
                             eos_token_id=None,
                             repetition_penalty=1.0,
                             length_penalty=1.0,
                             pad_token_id=0,
                             restricter=None):
    device = _get_model_device(decoder)
    if tokens is None:
        if bos_token_id is None:
            raise RuntimeError(
                "You have to specify either `tokens` or `bos_token_id`.")
        batch_size = state.num_samples
        if batch_size is None:
            raise RuntimeError(
                "Cannot infer the number of samples from `state`.")
        tokens = torch.full([batch_size, 1],
                            fill_value=bos_token_id,
                            dtype=torch.long).to(device)
    batch_size = tokens.size(0)
    if state.num_samples:
        assert state.num_samples == batch_size, "The number of samples in `tokens` and `state` should match."

    if eos_token_id is None:
        _eos_token_id = -1
    else:
        _eos_token_id = eos_token_id

    scores = decoder.decode(tokens=tokens, state=state)

    if restricter is not None:
        _, next_tokens = restricter(state, tokens, scores, num_beams=1)
    else:
        next_tokens = scores.argmax(dim=-1, keepdim=True)
    token_ids = torch.cat([tokens, next_tokens], dim=1)
    cur_len = token_ids.size(1)
    dones = token_ids.new_zeros(batch_size).eq(1).__or__(
        next_tokens.squeeze(1).eq(eos_token_id))
    # tokens = tokens[:, -1:]

    if max_len_a != 0:
        # (bsz x num_beams, )
        if state.encoder_mask is not None:
            max_lengths = (state.encoder_mask.sum(dim=1).float() *
                           max_len_a).long() + max_length
        else:
            max_lengths = tokens.new_full((tokens.size(0), ),
                                          fill_value=max_length,
                                          dtype=torch.long)
        real_max_length = max_lengths.max().item()
    else:
        real_max_length = max_length
        if state.encoder_mask is not None:
            max_lengths = state.encoder_mask.new_ones(
                state.encoder_mask.size(0)).long() * max_length
        else:
            max_lengths = tokens.new_full((tokens.size(0), ),
                                          fill_value=max_length,
                                          dtype=torch.long)

    while cur_len < real_max_length:
        scores = decoder.decode(tokens=token_ids,
                                state=state)  # batch_size x vocab_size

        if repetition_penalty != 1.0:
            token_scores = scores.gather(dim=1, index=token_ids)
            lt_zero_mask = token_scores.lt(0).float()
            ge_zero_mask = lt_zero_mask.eq(0).float()
            token_scores = lt_zero_mask * repetition_penalty * token_scores + ge_zero_mask / repetition_penalty * token_scores
            scores.scatter_(dim=1, index=token_ids, src=token_scores)

        if eos_token_id is not None and length_penalty != 1.0:
            token_scores = scores / cur_len**length_penalty  # batch_size x vocab_size
            eos_mask = scores.new_ones(scores.size(1))
            eos_mask[eos_token_id] = 0
            eos_mask = eos_mask.unsqueeze(0).eq(1)
            scores = scores.masked_scatter(
                eos_mask, token_scores)

        if restricter is not None:
            _, next_tokens = restricter(state, token_ids, scores, 1)
        else:
            next_tokens = scores.argmax(dim=-1, keepdim=True)
        next_tokens = next_tokens.squeeze(-1)

        if _eos_token_id != -1:
            next_tokens = next_tokens.masked_fill(max_lengths.eq(cur_len + 1),
                                                  _eos_token_id)
        next_tokens = next_tokens.masked_fill(
            dones, pad_token_id)
        tokens = next_tokens.unsqueeze(1)

        token_ids = torch.cat([token_ids, tokens],
                              dim=-1)  # batch_size x max_len

        end_mask = next_tokens.eq(_eos_token_id)
        dones = dones.__or__(end_mask)
        cur_len += 1

        if dones.min() == 1:
            break

    return token_ids


def sc_generate(decoder: Seq2SeqDecoder,
                state,
                tokens=None,
                gt_tokens=None,
                max_length=20,
                max_len_a=0.0,
                bos_token_id=None,
                eos_token_id=None,
                repetition_penalty=1.0,
                length_penalty=1.0,
                pad_token_id=0,
                restricter=None):
    device = _get_model_device(decoder)
    if tokens is None:
        if bos_token_id is None:
            raise RuntimeError("You have to specify either `tokens` or `bos_token_id`.")
        batch_size = state.num_samples
        if batch_size is None:
            raise RuntimeError(
                "Cannot infer the number of samples from `state`.")
        tokens = torch.full([batch_size, 1],
                            fill_value=bos_token_id,
                            dtype=torch.long).to(device)
    batch_size = tokens.size(0)
    # print(state.num_samples, batch_size)
    if state.num_samples:
        assert state.num_samples == batch_size, "The number of samples in `tokens` and `state` should match."

    if eos_token_id is None:
        _eos_token_id = -1
    else:
        _eos_token_id = eos_token_id
    aspect_cnt = 3
    next_tokens = gt_tokens[:, aspect_cnt:aspect_cnt + 2]
    token_ids = torch.cat([tokens, next_tokens], dim=1)
    cur_len = token_ids.size(1)
    dones = token_ids.new_zeros(batch_size).eq(1)
    # tokens = tokens[:, -1:]
    max_len_a = 0
    max_length = gt_tokens.size(1)
    gt_mask = gt_tokens.eq(1).eq(0)
    max_lengths = gt_mask.sum(dim=1)

    while cur_len < max_length:
        scores = decoder.decode(tokens=token_ids, state=state, only_sc=True)

        if restricter is not None:
            _, next_tokens = restricter(state, token_ids, scores, 1)
        else:
            next_tokens = scores.argmax(dim=-1, keepdim=True)
        next_tokens = next_tokens.squeeze(-1)

        next_tokens = next_tokens.masked_fill(
            dones, pad_token_id)  # 对已经搜索完成的sample做padding
        tokens = next_tokens.unsqueeze(1)

        token_ids = torch.cat([token_ids, tokens],
                              dim=-1)  # batch_size x max_len

        # end_mask = next_tokens.eq(_eos_token_id)
        # dones = dones.__or__(end_mask)
        dones = gt_tokens[:, cur_len + 1].eq(1)
        cur_len += 1
        aspect_cnt += 3
        if aspect_cnt + 2 < max_length:
            token_ids = torch.cat([token_ids, gt_tokens[:, aspect_cnt:aspect_cnt + 2]], dim=-1)
        cur_len += 2

        if dones.min() == 1:
            break
    ones = token_ids.new_ones(batch_size).unsqueeze(-1)
    token_ids = torch.cat([token_ids, ones], dim=-1)

    return token_ids


def _beam_search_generate(decoder: Seq2SeqDecoder,
                          tokens=None,
                          state=None,
                          max_length=20,
                          max_len_a=0.0,
                          num_beams=4,
                          bos_token_id=None,
                          eos_token_id=None,
                          do_sample=True,
                          repetition_penalty=1.0,
                          length_penalty=None,
                          pad_token_id=0,
                          restricter=None) -> torch.LongTensor:
    assert do_sample is False
    # 进行beam search
    device = _get_model_device(decoder)
    if tokens is None:
        if bos_token_id is None:
            raise RuntimeError(
                "You have to specify either `tokens` or `bos_token_id`.")
        batch_size = state.num_samples
        if batch_size is None:
            raise RuntimeError(
                "Cannot infer the number of samples from `state`.")
        tokens = torch.full([batch_size, 1],
                            fill_value=bos_token_id,
                            dtype=torch.long).to(device)
    batch_size = tokens.size(0)
    if state.num_samples:
        assert state.num_samples == batch_size, "The number of samples in `tokens` and `state` should match."

    if eos_token_id is None:
        _eos_token_id = -1
    else:
        _eos_token_id = eos_token_id

    scores = decoder.decode(tokens=tokens, state=state)

    vocab_size = scores.size(1)
    assert vocab_size >= num_beams, "num_beams should be smaller than the number of vocabulary size."

    scores = F.log_softmax(scores, dim=-1)  # (batch_size, vocab_size)

    if restricter is not None:
        _next_scores, _next_tokens = restricter(state, tokens, scores, num_beams + 1)
    else:
        _next_scores, _next_tokens = torch.topk(scores,
                                                num_beams + 1,
                                                dim=1,
                                                largest=True,
                                                sorted=True)

    indices = torch.arange(batch_size, dtype=torch.long).to(device)
    indices = indices.repeat_interleave(num_beams)
    state.reorder_state(indices)
    tokens = tokens.index_select(dim=0, index=indices)  # batch_size * num_beams x length

    if max_len_a != 0:
        # (bsz x num_beams, )
        if state.encoder_mask is not None:
            max_lengths = (state.encoder_mask.sum(dim=1).float() * max_len_a).long() + max_length
        else:
            max_lengths = tokens.new_full((batch_size * num_beams,),
                                          fill_value=max_length,
                                          dtype=torch.long)
        real_max_length = max_lengths.max().item()
    else:
        real_max_length = max_length
        if state.encoder_mask is not None:
            max_lengths = state.encoder_mask.new_ones(state.encoder_mask.size(0)).long() * max_length
        else:
            max_lengths = tokens.new_full((batch_size * num_beams,),
                                          fill_value=max_length,
                                          dtype=torch.long)
    hypos = [BeamHypotheses(num_beams, real_max_length, length_penalty, early_stopping=False) for _ in range(batch_size)]

    not_eos_mask = _next_tokens.ne(_eos_token_id)
    keep_mask = not_eos_mask.cumsum(dim=1).le(num_beams)
    keep_mask = not_eos_mask.__and__(keep_mask)

    next_tokens = _next_tokens.masked_select(keep_mask).view(batch_size, num_beams)
    next_scores = _next_scores.masked_select(keep_mask).view(batch_size, num_beams)

    rows, cols = not_eos_mask.eq(0)[:, :num_beams].nonzero(as_tuple=True)

    if len(rows) > 0:
        for row, col in zip(rows.tolist(), cols.tolist()):
            _token = torch.cat([tokens[row * num_beams], _next_tokens[row, col:col + 1]], dim=0)
            hypos[row].add(_token.clone(), _next_scores[row, col].item())

    # 记录生成好的token (batch_size', cur_len)
    token_ids = torch.cat([tokens, next_tokens.view(-1, 1)], dim=-1)
    dones = [False] * batch_size

    beam_scores = next_scores.view(-1)  # batch_size * num_beams

    cur_len = token_ids.size(1)

    # 0, num_beams, 2*num_beams, ...
    batch_inds_with_numbeams_interval = (torch.arange(batch_size) * num_beams).view(-1, 1).to(token_ids)

    while cur_len < real_max_length:
        scores = decoder.decode(token_ids, state)  # (bsz x num_beams, vocab_size)
        if repetition_penalty != 1.0:
            token_scores = scores.gather(dim=1, index=token_ids)
            lt_zero_mask = token_scores.lt(0).float()
            ge_zero_mask = lt_zero_mask.eq(0).float()
            token_scores = lt_zero_mask * repetition_penalty * token_scores + ge_zero_mask / repetition_penalty * token_scores
            scores.scatter_(dim=1, index=token_ids, src=token_scores)

        if _eos_token_id != -1:
            max_len_eos_mask = max_lengths.eq(cur_len + 1)
            eos_scores = scores[:, _eos_token_id]
            scores[:, _eos_token_id] = torch.where(max_len_eos_mask,
                                                   eos_scores + 1e32,
                                                   eos_scores)

        scores = F.log_softmax(scores, dim=-1)  # (batch_size * num_beams, vocab_size)
        _scores = scores + beam_scores[:, None]  # (batch_size * num_beams, vocab_size)
        _scores = _scores.view(batch_size, -1)  # (batch_size, num_beams*vocab_size)
        if restricter is not None:
            next_scores, ids = restricter(state, token_ids, _scores, 2 * num_beams)
        else:
            next_scores, ids = torch.topk(_scores,
                                          2 * num_beams,
                                          dim=1,
                                          largest=True,
                                          sorted=True)  # (bsz, 2*num_beams)
        from_which_beam = ids // vocab_size  # (batch_size, 2*num_beams)
        next_tokens = ids % vocab_size  # (batch_size, 2*num_beams)

        not_eos_mask = next_tokens.ne(_eos_token_id)
        keep_mask = not_eos_mask.cumsum(dim=1).le(num_beams)
        keep_mask = not_eos_mask.__and__(keep_mask)

        _next_tokens = next_tokens.masked_select(keep_mask).view(-1, 1)
        _from_which_beam = from_which_beam.masked_select(keep_mask).view(batch_size, num_beams)
        _next_scores = next_scores.masked_select(keep_mask).view(batch_size, num_beams)
        beam_scores = _next_scores.view(-1)

        flag = True
        if cur_len + 1 == real_max_length:
            eos_batch_idx = torch.arange(batch_size).to(next_tokens).repeat_interleave(repeats=num_beams, dim=0)
            eos_beam_ind = torch.arange(num_beams).to(token_ids).repeat(batch_size)
            eos_beam_idx = from_which_beam[:, :num_beams].reshape(-1)
        else:
            effective_eos_mask = next_tokens[:, :num_beams].eq(_eos_token_id)  # batch_size x num_beams
            if effective_eos_mask.sum().gt(0):
                eos_batch_idx, eos_beam_ind = effective_eos_mask.nonzero(as_tuple=True)
                eos_beam_idx = eos_batch_idx * num_beams * 2 + eos_beam_ind
                eos_beam_idx = from_which_beam.view(-1)[eos_beam_idx]
            else:
                flag = False

        if flag:
            _token_ids = torch.cat([token_ids, _next_tokens], dim=-1)
            for batch_idx, beam_ind, beam_idx in zip(eos_batch_idx.tolist(), eos_beam_ind.tolist(), eos_beam_idx.tolist()):
                if not dones[batch_idx]:
                    score = next_scores[batch_idx, beam_ind].item()
                    if _eos_token_id != -1:
                        hypos[batch_idx].add(_token_ids[batch_idx * num_beams + beam_idx, :cur_len].clone(), score)
                    else:
                        hypos[batch_idx].add( _token_ids[batch_idx * num_beams + beam_idx].clone(), score)

        reorder_inds = (batch_inds_with_numbeams_interval + _from_which_beam).view(-1)
        state.reorder_state(reorder_inds)
        token_ids = torch.cat([token_ids.index_select(index=reorder_inds, dim=0), _next_tokens], dim=-1)

        for batch_idx in range(batch_size):
            dones[batch_idx] = dones[batch_idx] or hypos[batch_idx].is_done(next_scores[batch_idx, 0].item()) or \
                               max_lengths[batch_idx*num_beams]==cur_len+1

        cur_len += 1

        if all(dones):
            break

    # select the best hypotheses
    tgt_len = token_ids.new_zeros(batch_size)
    best = []

    for i, hypotheses in enumerate(hypos):
        best_hyp = max(hypotheses.hyp, key=lambda x: x[0])[1]
        # 把上面替换为非eos的词替换回eos
        if _eos_token_id != -1:
            best_hyp = torch.cat([best_hyp, best_hyp.new_ones(1) * _eos_token_id])
        tgt_len[i] = len(best_hyp)
        best.append(best_hyp)

    # generate target batch
    decoded = token_ids.new_zeros(batch_size, tgt_len.max().item()).fill_(pad_token_id)
    for i, hypo in enumerate(best):
        decoded[i, :tgt_len[i]] = hypo

    return decoded


class BeamHypotheses(object):
    def __init__(self, num_beams, max_length, length_penalty, early_stopping):
        """
        Initialize n-best list of hypotheses.
        """
        self.max_length = max_length - 1  # ignoring bos_token
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.num_beams = num_beams
        self.hyp = []
        self.worst_score = 1e9

    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.hyp)

    def add(self, hyp, sum_logprobs):
        """
        Add a new hypothesis to the list.
        """
        score = sum_logprobs / len(hyp)**self.length_penalty
        if len(self) < self.num_beams or score > self.worst_score:
            self.hyp.append((score, hyp))
            if len(self) > self.num_beams:
                sorted_scores = sorted([(s, idx) for idx, (s, _) in enumerate(self.hyp)])
                del self.hyp[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs):
        """
        If there are enough hypotheses and that none of the hypotheses being generated
        can become better than the worst one in the heap, then we are done with this sentence.
        """
        if len(self) < self.num_beams:
            return False
        elif self.early_stopping:
            return True
        else:
            return self.worst_score >= best_sum_logprobs / self.max_length**self.length_penalty
