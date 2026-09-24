"""Checks for the dynamic N-grams task's data (paper, Section 4.4).

Three things can go wrong here that the shapes will not catch: the sequence could be sampled
from the wrong entry of the table, the prior could be the wrong one, and the optimal estimator
could be off by a step. So the checks sample from hand-made tables whose statistics are known,
and compare the estimator both against Equation 10 worked out by hand and against chance.
"""

import torch

from src.tasks.dynamic_ngrams import data


def test_shapes_match_the_declared_channel_counts():
    x, target, mask = data.batch(2)

    assert x.shape == (2, data.LENGTH, data.INPUT_SIZE) == (2, 200, 1)
    assert target.shape == (2, data.LENGTH, data.OUTPUT_SIZE) == (2, 200, 1)
    assert mask.shape == (2, 200)
    assert data.TIMESTEPS == (200,)  # one length, so one CUDA graph
    assert set(target.flatten().tolist()) <= {0.0, 1.0}


def test_the_input_is_the_target_delayed_by_one_step():
    x, target, _ = data.batch(4)

    assert x[:, 0, 0].sum() == 0  # nothing has been seen yet at t = 0
    assert torch.equal(x[:, 1:, 0], target[:, :-1, 0])  # every other step sees the bit just scored


def test_the_mask_excludes_exactly_the_first_five_steps():
    _, _, mask = data.batch(4)

    assert not mask[:, :data.CONTEXT].any()  # drawn i.i.d., with no context to predict from
    assert mask[:, data.CONTEXT:].all()
    assert int(mask[0].sum()) == 195


def test_short_sequences_still_have_something_to_score():
    x, _, mask = data.batch(2, min_len=3, max_len=3)  # what the demo asks for

    assert x.shape[1] == data.CONTEXT + 1  # floored, so one step is scored
    assert int(mask[0].sum()) == 1


def test_the_tables_come_from_the_beta_prior():
    torch.manual_seed(0)
    tables = data.sample_tables(400)

    assert tables.shape == (400, 32)
    assert 0.48 < tables.mean() < 0.52
    # Beta(1/2, 1/2) is U-shaped: its CDF puts exactly a third of the mass below 0.25, where a
    # uniform prior would put a quarter.
    assert abs((tables < 0.25).float().mean() - 1 / 3) < 0.03
    assert abs((tables > 0.75).float().mean() - 1 / 3) < 0.03


def test_a_lopsided_table_gives_lopsided_bits():
    torch.manual_seed(0)
    bits = data.sample_bits(torch.full((64, data.TABLE_SIZE), 0.9))

    # Every context says "90% ones", so the bits drawn with a context should be about 90% ones.
    # 64 * 195 draws, so the tolerance is far wider than the sampling noise.
    assert abs(bits[:, data.CONTEXT:].mean() - 0.9) < 0.02
    assert abs(bits[:, :data.CONTEXT].mean() - 0.5) < 0.1  # the first five ignore the table


def test_the_bits_are_sampled_from_the_entry_their_history_points_at():
    # A table that says "the next bit repeats the oldest bit of the context": the top half of the
    # table, where that bit is one, is certain of a one, and the bottom half is certain of a zero.
    table = (torch.arange(data.TABLE_SIZE) >= data.TABLE_SIZE // 2).float()
    bits = data.sample_bits(table.expand(8, -1))

    assert torch.equal(bits[:, data.CONTEXT:], bits[:, :-data.CONTEXT])


def test_the_optimal_estimator_is_equation_10():
    bits = torch.zeros(1, 12)  # one context, seen over and over, followed only by zeros
    predictions = data.optimal_predictions(bits)

    assert predictions[0, :data.CONTEXT].tolist() == [0.5] * 5  # no context, so chance
    # N1 = 0 and N0 grows by one each step, so Equation 10 gives (0 + 1/2) / (N0 + 1)
    expected = [0.5 / (seen + 1) for seen in range(12 - data.CONTEXT)]
    assert torch.allclose(predictions[0, data.CONTEXT:], torch.tensor(expected))


def test_the_optimal_estimator_beats_chance_by_a_wide_margin():
    torch.manual_seed(0)
    _, target, mask = data.batch(200)
    bits = target[:, :, 0]

    optimal = data.cost_in_bits(data.optimal_predictions(bits), bits, mask) / mask.sum(-1)
    chance = data.cost_in_bits(torch.full_like(bits, 0.5), bits, mask) / mask.sum(-1)

    assert chance.mean() == 1.0  # a coin flip costs one bit a step, by definition
    assert optimal.mean() < 0.85  # measured at about 0.68 bits a step, near the paper's Figure 13
    # The estimator knows the prior but not the table, so it cannot beat the conditional entropy
    # of a Beta(1/2, 1/2) bit, which is (2 ln 2 - 1) / ln 2 = 0.557 bits. Anything below that
    # means the predictions have been peeked at the table or shifted onto the bit they predict.
    assert optimal.mean() > 0.55
