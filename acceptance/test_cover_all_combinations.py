import pytest

from sweetpea import *
from sweetpea._internal.constraint import CoverAllCombinations


def _stroop(n):
    """Build a Stroop-style inner block with n colors/words and a derived congruency
    factor, crossing [congruency, color] (word left free)."""
    colors = ['red', 'green', 'blue', 'yellow', 'purple'][:n]
    color = Factor('color', colors)
    word = Factor('word', colors)
    congruency = Factor('congruency', [
        DerivedLevel('con', WithinTrial(lambda c, w: c == w, [color, word])),
        DerivedLevel('incon', WithinTrial(lambda c, w: c != w, [color, word])),
    ])
    inner = CrossBlock([congruency, color, word], [congruency, color], [])
    return colors, color, word, congruency, inner


def _stroop_nest(n, extra_constraints=[]):
    colors, color, word, congruency, inner = _stroop(n)
    instance = Factor('instance', ['a', 'b'])
    outer = CrossBlock([instance], [instance], [])
    nest = Nest(outer, inner,
                [CoverAllCombinations(color, word)] + extra_constraints)
    return colors, color, word, nest


def _covers_all(experiment, colors):
    all_pairs = set((c, w) for c in colors for w in colors)
    return set(zip(experiment['color'], experiment['word'])) == all_pairs


# ~~~~~~~~~~~~ K computation (isolated) ~~~~~~~~~~~~

@pytest.mark.parametrize('n,expected_k', [(3, 2), (4, 3), (5, 4)])
def test_required_instances_stroop(n, expected_k):
    _, color, word, _, inner = _stroop(n)
    assert CoverAllCombinations(color, word).required_instances(inner) == expected_k


def test_required_instances_overlap_below_biggest_group():
    # Two free factors, neither crossed: slots fully overlap, so K = ceil(4/2) = 2,
    # not the naive "biggest group" count of 4.
    colr = Factor('colr', ['red', 'green'])
    size = Factor('size', ['big', 'small'])
    task = Factor('task', ['A', 'B'])
    inner = CrossBlock([task, colr, size], [task], [])
    assert CoverAllCombinations(colr, size).required_instances(inner) == 2


# ~~~~~~~~~~~~ Auto-sizing + coverage (Nest, end to end) ~~~~~~~~~~~~

@pytest.mark.parametrize('n,expected_trials', [(3, 12), (4, 32)])
def test_autosize_trial_count(n, expected_trials):
    _, _, _, nest = _stroop_nest(n)
    assert nest.trials_per_sample() == expected_trials


@pytest.mark.parametrize('n', [3, 4])
def test_coverage_holds(n):
    colors, _, _, nest = _stroop_nest(n)
    exps = synthesize_trials(nest, 5, sampling_strategy=IterateGen)
    assert exps
    for e in exps:
        assert _covers_all(e, colors)


# ~~~~~~~~~~~~ Coverage on other constructs ~~~~~~~~~~~~

def test_plain_crossblock_coverage_with_minimum_trials():
    # A plain CrossBlock with an uncrossed factor, sized by the user via
    # MinimumTrials, must still achieve coverage.
    colors, color, word, congruency, _ = _stroop(3)
    block = CrossBlock([congruency, color, word], [congruency, color],
                       [CoverAllCombinations(color, word), MinimumTrials(12)])
    exps = synthesize_trials(block, 3, sampling_strategy=IterateGen)
    assert exps
    for e in exps:
        assert _covers_all(e, colors)


def test_plain_crossblock_autosizes():
    # No Nest and no MinimumTrials: the CrossBlock grows itself to fit coverage.
    colors, color, word, congruency, _ = _stroop(3)
    block = CrossBlock([congruency, color, word], [congruency, color],
                       [CoverAllCombinations(color, word)])
    assert block.trials_per_sample() == 12
    exps = synthesize_trials(block, 3, sampling_strategy=IterateGen)
    assert exps
    for e in exps:
        assert _covers_all(e, colors)


def test_merge_autosizes():
    # Coverage attached to a Merge: two fully-free factors, K = ceil(4/2) = 2
    # passes of the [task] crossing -> 4 trials.
    colr = Factor('colr', ['red', 'green'])
    size = Factor('size', ['big', 'small'])
    task = Factor('task', ['A', 'B'])
    base = CrossBlock([task, colr, size], [task], [])
    block = Merge([base], [CoverAllCombinations(colr, size)])
    assert block.trials_per_sample() == 4
    exps = synthesize_trials(block, 3, sampling_strategy=IterateGen)
    assert exps
    all_pairs = set((c, s) for c in ['red', 'green'] for s in ['big', 'small'])
    for e in exps:
        assert set(zip(e['colr'], e['size'])) == all_pairs


def test_repeat_autosizes():
    colors, color, word, congruency, _ = _stroop(3)
    base = CrossBlock([congruency, color, word], [congruency, color], [])
    block = Repeat(base, [CoverAllCombinations(color, word)])
    assert block.trials_per_sample() == 12
    exps = synthesize_trials(block, 3, sampling_strategy=IterateGen)
    assert exps
    for e in exps:
        assert _covers_all(e, colors)


# ~~~~~~~~~~~~ Interaction with other constraints ~~~~~~~~~~~~

def test_outer_factor_listed_constructs():
    # An outer-block factor in the listed set is not statically sized; the
    # solver arbitrates. Construction must succeed.
    _, color, word, _, inner = _stroop(3)
    instance = Factor('instance', ['a', 'b'])
    outer = CrossBlock([instance], [instance], [])
    Nest(outer, inner, [CoverAllCombinations(instance, color)])


def test_sequential_on_listed_factor():
    # Sequential pins word per position; positional sizing finds K=2 (12
    # trials), and synthesis must produce covering sequences.
    colors, color, word, congruency, inner = _stroop(3)
    instance = Factor('instance', ['a', 'b'])
    outer = CrossBlock([instance], [instance], [])
    nest = Nest(outer, inner, [CoverAllCombinations(color, word), Sequential(word)])
    assert nest.trials_per_sample() == 12
    exps = synthesize_trials(nest, 2, sampling_strategy=IterateGen)
    assert exps
    for e in exps:
        assert _covers_all(e, colors)
        # Sequential holds: word cycles through its levels in order.
        assert e['word'] == (colors * 4)[:len(e['word'])]


def test_latin_square_on_listed_factors():
    # LatinSquare and CoverAllCombinations over the same factors: the solver
    # satisfies both (LatinSquare's rotations themselves yield full coverage).
    colors, color, word, congruency, inner = _stroop(3)
    instance = Factor('instance', ['a', 'b'])
    outer = CrossBlock([instance], [instance], [])
    nest = Nest(outer, inner, [LatinSquare([color, word], name='P'),
                               CoverAllCombinations(color, word)])
    exps = synthesize_trials(nest, 1, sampling_strategy=IterateGen)
    for e in exps:
        assert _covers_all(e, colors)


# ~~~~~~~~~~~~ Reconciliation ~~~~~~~~~~~~

def test_exclude_shrinks_required_set():
    # An excluded level's combinations are dropped from the required set rather
    # than making coverage unsatisfiable.
    colr = Factor('colr', ['red', 'green'])
    size = Factor('size', ['big', 'small'])
    task = Factor('task', ['A', 'B'])
    block = CrossBlock([task, colr, size], [task],
                       [CoverAllCombinations(colr, size), Exclude((colr, 'red'))])
    assert block.trials_per_sample() == 2  # only 2 combos left to cover
    exps = synthesize_trials(block, 3, sampling_strategy=IterateGen)
    assert exps
    for e in exps:
        assert set(zip(e['colr'], e['size'])) == {('green', 'big'), ('green', 'small')}


def test_exactly_k_below_coverage_errors():
    # Coverage needs 3 word-red trials (red-red forced, green-red, blue-red);
    # ExactlyK(1) can never be reconciled by adding instances -> named error.
    colors, color, word, congruency, _ = _stroop(3)
    with pytest.raises(ValueError):
        CrossBlock([congruency, color, word], [congruency, color],
                   [CoverAllCombinations(color, word), ExactlyK(1, (word, 'red'))])


def test_exactly_k_reconciled():
    # ExactlyK(4): 2 instances contribute 2 forced (red,red) + the 2 free
    # word-red combos = exactly 4. K stays 2 -> 12 trials.
    colors, color, word, congruency, _ = _stroop(3)
    block = CrossBlock([congruency, color, word], [congruency, color],
                       [CoverAllCombinations(color, word), ExactlyK(4, (word, 'red'))])
    assert block.trials_per_sample() == 12
    exps = synthesize_trials(block, 2, sampling_strategy=IterateGen)
    assert exps
    for e in exps:
        assert _covers_all(e, colors)
        assert e['word'].count('red') == 4


def test_pin_grows_k():
    # A Pin on a listed factor conservatively consumes one free pick, growing
    # K from 2 to 3 (18 trials); coverage and the pin both hold.
    colors, color, word, congruency, _ = _stroop(3)
    block = CrossBlock([congruency, color, word], [congruency, color],
                       [CoverAllCombinations(color, word), Pin(0, (word, 'red'))])
    assert block.trials_per_sample() == 18
    exps = synthesize_trials(block, 2, sampling_strategy=IterateGen)
    assert exps
    for e in exps:
        assert _covers_all(e, colors)
        assert e['word'][0] == 'red'


# ~~~~~~~~~~~~ Analytic K lower bound ~~~~~~~~~~~~

@pytest.mark.parametrize('n,expected_k', [(3, 2), (4, 3), (5, 4)])
def test_k_lower_bound_tight_for_stroop(n, expected_k):
    # The analytic floor equals the true K for the standard cases, so the
    # matching scan does a single confirming check.
    _, color, word, _, inner = _stroop(n)
    cac = CoverAllCombinations(color, word)
    (_, _, R_free, slots, _, sw, _) = cac._coverage_analysis(inner, [color, word])
    assert cac._k_lower_bound(R_free, slots, sw) == expected_k
    assert cac.required_instances(inner) == expected_k


def test_k_lower_bound_scan_corrects_upward():
    # An overlapping-union case the signature bounds don't capture: slots 0 and 1
    # jointly hold {a, b, c} (3 demands, 2 picks/instance) but no single signature
    # nor the global bound sees it (slot 2's weight inflates global capacity).
    slots = [{'a', 'c'}, {'b', 'c'}, {'d'}]
    weights = [1, 1, 4]
    R_free = {'a', 'b', 'c', 'd'}
    assert CoverAllCombinations._k_lower_bound(R_free, slots, weights) == 1
    assert CoverAllCombinations._min_instances(R_free, slots, weights) == 2


# ~~~~~~~~~~~~ Weighted balances ~~~~~~~~~~~~

def _stroop_weighted(n, incon_weight):
    """Stroop with a weighted congruency ratio: 1 congruent : `incon_weight`
    incongruent trials per color, per pass."""
    colors = ['red', 'green', 'blue', 'yellow'][:n]
    color = Factor('color', colors)
    word = Factor('word', colors)
    congruency = Factor('congruency', [
        DerivedLevel('con', WithinTrial(lambda c, w: c == w, [color, word]), 1),
        DerivedLevel('incon', WithinTrial(lambda c, w: c != w, [color, word]), incon_weight),
    ])
    return colors, color, word, congruency


@pytest.mark.parametrize('incon_weight,expected_k', [(1, 3), (2, 2), (3, 1)])
def test_required_instances_weighted(incon_weight, expected_k):
    colors, color, word, congruency = _stroop_weighted(4, incon_weight)
    inner = CrossBlock([congruency, color, word], [congruency, color], [])
    assert CoverAllCombinations(color, word).required_instances(inner) == expected_k


@pytest.mark.parametrize('incon_weight,expected_trials,expected_con,expected_incon',
                         [(2, 24, 8, 16),   # K=2 passes of 12 -> 1:2 overall
                          (3, 16, 4, 12)])  # natural ratio: one pass suffices
def test_weighted_autosize_and_coverage(incon_weight, expected_trials,
                                        expected_con, expected_incon):
    colors, color, word, congruency = _stroop_weighted(4, incon_weight)
    block = CrossBlock([congruency, color, word], [congruency, color],
                       [CoverAllCombinations(color, word)])
    assert block.trials_per_sample() == expected_trials
    exps = synthesize_trials(block, 2, sampling_strategy=IterateGen)
    assert exps
    for e in exps:
        assert _covers_all(e, colors)
        assert e['congruency'].count('con') == expected_con
        assert e['congruency'].count('incon') == expected_incon


def test_unmodeled_conflict_yields_hint(capsys):
    # In-a-row constraints aren't statically reconciled; a truly unsatisfiable
    # combination yields 0 samples plus a printed hint (no crash).
    colors, color, word, congruency, _ = _stroop(3)
    block = CrossBlock([congruency, color, word], [congruency, color],
                       [CoverAllCombinations(color, word),
                        AtLeastKInARow(7, (word, 'red'))])
    exps = synthesize_trials(block, 1, sampling_strategy=IterateGen)
    assert exps == []
    out = capsys.readouterr().out
    assert 'CoverAllCombinations' in out
    assert 'AtLeastKInARow' in out


# ~~~~~~~~~~~~ Coverage/trials tradeoff (prioritize) ~~~~~~~~~~~~

def _trio():
    """`task` is crossed while colr, size, and cue ride free, so covering all
    three together is what drives the trial count: 2*2*3 = 12 combinations over
    2 free slots per pass, hence 12 trials."""
    colr = Factor('colr', ['red', 'green'])
    size = Factor('size', ['big', 'small'])
    cue = Factor('cue', ['c1', 'c2', 'c3'])
    task = Factor('task', ['A', 'B'])
    return colr, size, cue, task


def _trio_block(colr, size, cue, task, listed=None, **kw):
    listed = listed if listed is not None else [colr, size, cue]
    return CrossBlock(design=[task, colr, size, cue], crossing=[task],
                      constraints=[CoverAllCombinations(*listed, **kw)])


class _Answers:
    """Stands in for `input`, replaying scripted answers and recording the
    prompts it was asked."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.prompts = []

    def __call__(self, prompt=''):
        self.prompts.append(prompt)
        if not self.answers:
            raise AssertionError('unexpected extra prompt: {}'.format(prompt))
        return self.answers.pop(0)


@pytest.mark.parametrize('kw', [{}, {'prioritize': False}])
def test_prioritize_off_keeps_full_coverage(kw):
    colr, size, cue, task = _trio()
    assert _trio_block(colr, size, cue, task, **kw).trials_per_sample() == 12


def test_prioritize_reduces_trials():
    # colr x size needs 2 passes and cue's 3 levels need 2, so K = 2.
    colr, size, cue, task = _trio()
    block = _trio_block(colr, size, cue, task, prioritize=[colr, size])
    assert block.trials_per_sample() == 4


def test_prioritize_every_factor_is_full_coverage():
    colr, size, cue, task = _trio()
    block = _trio_block(colr, size, cue, task, prioritize=[colr, size, cue])
    assert block.trials_per_sample() == 12


def test_prioritize_empty_demotes_everything():
    colr, size, cue, task = _trio()
    block = _trio_block(colr, size, cue, task, prioritize=[])
    assert block.trials_per_sample() == 4


def test_prioritize_ignores_duplicates():
    colr, size, cue, task = _trio()
    block = _trio_block(colr, size, cue, task, prioritize=[colr, colr, size])
    assert block.trials_per_sample() == 4


def test_prioritize_demoting_crossed_factor_is_harmless():
    # `task` is crossed, so its singleton group is satisfied by every pass.
    colr, size, cue, task = _trio()
    block = _trio_block(colr, size, cue, task, listed=[task, colr, size, cue],
                        prioritize=[colr, size])
    assert block.trials_per_sample() == 4


def test_prioritize_keeps_both_guarantees():
    colr, size, cue, task = _trio()
    block = _trio_block(colr, size, cue, task, prioritize=[colr, size])
    exps = synthesize_trials(block, 1, sampling_strategy=IterateGen)
    assert exps
    for e in exps:
        assert set(zip(e['colr'], e['size'])) == set(
            (c, s) for c in ['red', 'green'] for s in ['big', 'small'])
        assert set(e['cue']) == {'c1', 'c2', 'c3'}


def test_two_coverage_constraints_take_the_larger():
    # Two constraints express the same thing as prioritize=[colr, size].
    colr, size, cue, task = _trio()
    block = CrossBlock(design=[task, colr, size, cue], crossing=[task],
                       constraints=[CoverAllCombinations(colr, size),
                                    CoverAllCombinations(cue)])
    assert block.trials_per_sample() == 4
    exps = synthesize_trials(block, 1, sampling_strategy=IterateGen)
    assert exps
    for e in exps:
        assert set(zip(e['colr'], e['size'])) == set(
            (c, s) for c in ['red', 'green'] for s in ['big', 'small'])
        assert set(e['cue']) == {'c1', 'c2', 'c3'}


def test_prioritize_unknown_factor_raises():
    colr, size, cue, task = _trio()
    with pytest.raises(ValueError):
        CoverAllCombinations(colr, size, prioritize=[cue])


def test_prioritize_repr_shows_the_choice():
    colr, size, cue, task = _trio()
    assert repr(CoverAllCombinations(colr, size, cue, prioritize=[colr, size])) == \
        'CoverAllCombinations(colr, size, cue, prioritize=[colr, size])'
    assert repr(CoverAllCombinations(colr, size, cue, prioritize=True)) == \
        'CoverAllCombinations(colr, size, cue, prioritize=True)'


def test_prioritize_in_nest():
    colors, color, word, congruency, _ = _stroop(3)
    cue = Factor('cue', ['c1', 'c2', 'c3', 'c4'])
    inner = CrossBlock([congruency, color, word, cue], [congruency, color], [])
    instance = Factor('instance', ['a', 'b'])
    outer = CrossBlock([instance], [instance], [])
    # Full coverage: each incongruent cell is the only one that can serve its
    # own 8 word-cue combinations, so K = 8 over passes of 6.
    full = Nest(outer, inner, [CoverAllCombinations(color, word, cue)])
    assert full.trials_per_sample() == 48
    # Demoting cue leaves the 6 free color-word pairs over 3 slots: K = 2.
    fewer = Nest(outer, inner,
                 [CoverAllCombinations(color, word, cue, prioritize=[color, word])])
    assert fewer.trials_per_sample() == 12


def test_prioritize_with_weighted_crossing():
    # Weighted levels on the crossed factor still size correctly once some of
    # the listed factors are demoted.
    colors, color, word, congruency = _stroop_weighted(4, 2)
    cue = Factor('cue', ['c1', 'c2', 'c3'])
    block = CrossBlock([congruency, color, word, cue], [congruency, color],
                       [CoverAllCombinations(color, word, cue,
                                             prioritize=[color, word])])
    exps = synthesize_trials(block, 1, sampling_strategy=IterateGen)
    assert exps
    for e in exps:
        assert _covers_all(e, colors)
        assert set(e['cue']) == {'c1', 'c2', 'c3'}


# ~~~~~~~~~~~~ Interactive negotiation ~~~~~~~~~~~~

def test_interactive_accepting_the_offer_changes_nothing(monkeypatch):
    colr, size, cue, task = _trio()
    answers = _Answers('y')
    monkeypatch.setattr('builtins.input', answers)
    block = _trio_block(colr, size, cue, task, prioritize=True)
    assert block.trials_per_sample() == 12
    assert '12 trials' in answers.prompts[0]


def test_interactive_declining_then_choosing(monkeypatch, capsys):
    colr, size, cue, task = _trio()
    monkeypatch.setattr('builtins.input', _Answers('n', 'colr, size', 'y'))
    block = _trio_block(colr, size, cue, task, prioritize=True)
    assert block.trials_per_sample() == 4
    out = capsys.readouterr().out
    assert 'That gives 4 trials' in out
    assert 'cue' in out
    # The session has to be reproducible without the prompt.
    assert 'prioritize=[colr, size]' in out


def test_interactive_reprompts_on_unknown_factor(monkeypatch, capsys):
    colr, size, cue, task = _trio()
    monkeypatch.setattr('builtins.input', _Answers('n', 'task', 'colr, size', 'y'))
    block = _trio_block(colr, size, cue, task, prioritize=True)
    assert block.trials_per_sample() == 4
    assert 'Not listed in this constraint: task' in capsys.readouterr().out


def test_interactive_single_factor_skips_the_prompt(monkeypatch):
    # Nothing to demote, so the user is never asked.
    colr, size, cue, task = _trio()
    monkeypatch.setattr('builtins.input', _Answers())
    block = _trio_block(colr, size, cue, task, listed=[cue], prioritize=True)
    assert block.trials_per_sample() == 4


def test_prioritize_without_interactive_input_warns(monkeypatch):
    colr, size, cue, task = _trio()

    def _no_input(prompt=''):
        raise OSError('reading from stdin while output is captured')

    monkeypatch.setattr('builtins.input', _no_input)
    with pytest.warns(UserWarning, match='no interactive input'):
        block = _trio_block(colr, size, cue, task, prioritize=True)
    assert block.trials_per_sample() == 12


# ~~~~~~~~~~~~ Validation errors ~~~~~~~~~~~~

def test_error_weighted_levels():
    colors = ['red', 'green', 'blue']
    color = Factor('color', colors)
    word = Factor('word', [Level('red', 2), Level('green'), Level('blue')])
    congruency = Factor('congruency', [
        DerivedLevel('con', WithinTrial(lambda c, w: c == w, [color, word])),
        DerivedLevel('incon', WithinTrial(lambda c, w: c != w, [color, word])),
    ])
    inner = CrossBlock([congruency, color, word], [congruency, color], [])
    instance = Factor('instance', ['a', 'b'])
    outer = CrossBlock([instance], [instance], [])
    with pytest.raises(ValueError):
        Nest(outer, inner, [CoverAllCombinations(color, word)])
