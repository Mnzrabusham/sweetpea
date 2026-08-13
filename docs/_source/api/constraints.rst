.. _constraints:

Constraints
===========

.. class:: sweetpea.Constraint()

   Abstract class representing a constraint.
           

.. class:: sweetpea.Exclude(level)

              Constrains an experiment to disallow the specified
              level.

              An :class:`.Exclude` constraint can affect the number of
              trials that are included in a sequence. See
              :class:`.CrossBlock` for more information.

              :param level: either a level,
                            a tuple containing a factor and the name of one of its levels,
                            or a tuple containing a factor and one of its levels
              :type level: Union[Level, Tuple[Factor, Any], Tuple[Factor, Level]]
              :rtype: Constraint

.. class:: sweetpea.Pin(index, level)

              Constrains an experiment to require the specified level
              at the specified trial index. A negative trial index
              refers to a trial releative to the end of a sequence;
              for example, -1 refers to the last trial. If `index` is
              not in range for trials in an experiment, then the
              experiment will have no satisfying trial sequences.

              :param index: a trial index, counting forward from 0 or backward from -1
              :type index: int
              :param level: either a level,
                            a tuple containing a factor and the name of one of its levels,
                            or a tuple containing a factor and one of its levels
              :type level: Union[Level, Tuple[Factor, Any], Tuple[Factor, Level]]
              :rtype: Constraint

.. class:: sweetpea.MinimumTrials(k)

              Constrains an experiment to set the specified number of
              minimum trials. See :class:`.CrossBlock` and
              :class:`.Repeat` for more information.

              :param k: minimum number of trials
              :type k: int

.. class:: sweetpea.AtMostKInARow(k, level)

              Constrains an experiment to allow at most `k`
              consecutive trials with the level identified by
              `level`.

              :param k: the maximum number of consecutive repetitions
                        to allow
              :type k: int
              :param level: either a level,
                            a tuple containing a factor and the name of one of its levels,
                            a tuple containing a factor and one of its levels,
                            or just a factor; the last case is a shorthand for a separate
                            constraint for each of the factor's levels
              :type level: Union[Level, Tuple[Factor, Any], Tuple[Factor, Level], Factor]
              :rtype: Constraint

.. class:: sweetpea.AtLeastKInARow(k, level)

              Constrains an experiment so that when the level
              identified by `level` appears in a trial, it
              also appears in at least `k`-1 adjacent trials.
              
              :param k: the minimum number of consecutive repetitions
                        to require
              :type k: int
              :param level: like :class:`.AtMostKInARow`
              :type level: Union[Level, Tuple[Factor, Any], Tuple[Factor, Level], Factor]
              :rtype: Constraint

.. class:: sweetpea.ExactlyKInARow(k, level)

              Constrains an experiment so that when the level
              identified by `level` appears in a trial, it also
              appears in exactly `k`-1 adjacent trials.

              :param k: the number of repetitions to require
              :type k: int
              :param level: like :class:`.AtMostKInARow`
              :type level: Union[Level, Tuple[Factor, Any], Tuple[Factor, Level], Factor]
              :rtype: Constraint

.. class:: sweetpea.ExactlyK(k, level)

              Constrains an experiment so that the level identified by
              `level` appears in exactly `k` trials. If this
              constraint is not consistent with requirements for
              crossing, the experiment will have no satisfying trial
              sequences.

              :param k: the number of repetitions to require
              :type k: int
              :param level: like :class:`.AtMostKInARow`
              :type level: Union[Level, Tuple[Factor, Any], Tuple[Factor, Level], Factor]
              :rtype: Constraint

.. class:: sweetpea.Sequential(factor)

              Constrains the experiment so that the levels of `factor`
              must appear in order over a sequence of trials. When the
              last level of `factor` is used for a trial, the next
              trial starts the sequence again with the first level of
              `factor`.

              :param factor: the factor to constrain to seqential use of its
                             levels in an experiment
              :type factor: Factor
              :rtype: Constraint
           
.. class:: sweetpea.LatinSquare(factors)

              Constrains an experiment so that the levels of `factors`
              are combined in a Latin Square pattern. If the factor in
              `factors` with the most levels has N levels, then every
              N trials will include every level of every factor in
              `factors`. Furthermore, each subsequent sequence of N
              trials will have a distinct possible combination of
              levels until all possibilities are exhausted, and the
              combination order is deterministic.

              The given `factors` are typically crossed in an
              experiment description, but they are not required to be
              crossed explicitly; a :class:`LatinSquare` constraint
              effectively forces a crossing as long as an experiment
              includes enough trials.

              See :ref:`Latin Square Counterbalancing <latin-square-counterbalancing>`
              for more information.

              :param factors: the factors forming the Latin Square pattern
              :type factors: List[Factor]
              :rtype: Constraint

.. class:: sweetpea.CoverAllCombinations(*factors, prioritize=False)

              Constrains an experiment so that its trials collectively
              include every realizable combination of the levels of
              `factors` at least once. A factor that is left out of a
              crossing is otherwise assigned freely by the solver, so
              nothing normally guarantees that a particular combination
              ever appears; this constraint coordinates those free
              choices.

              Unlike most constraints, :class:`CoverAllCombinations`
              can increase the number of trials: the count needed for
              coverage is computed when the block is constructed, and
              the block grows to fit, rounded up so that every crossing
              still gets complete passes.

              A combination that cannot occur is not required. That
              includes combinations removed by an :class:`.Exclude`
              constraint and combinations that contradict the predicate
              of a :class:`.DerivedLevel`.

              The trial count reconciles the effects of :class:`.Pin`,
              :class:`.ExactlyK`, and :class:`.Sequential` constraints
              on the listed factors. Ordering constraints such as
              :class:`.AtMostKInARow` and :class:`.LatinSquare` are left
              to the solver instead; when one of those conflicts with
              coverage, :func:`.synthesize_trials` finds no sequences
              and reports which constraints were not accounted for.

              Levels of the listed factors must be unweighted, but
              weighted levels elsewhere in the crossing are supported,
              and they change the trial count accordingly.

              Since the combinations to cover are the product of the
              listed factors' levels, `prioritize` trades coverage for
              a shorter experiment. Given a list of the listed factors,
              those stay fully crossed, while each remaining factor is
              demoted to needing only each of its own levels present.
              Passing ``True`` instead reports the trial count as the
              block is built and asks which factors to keep; where no
              interactive input is available, that prompt is skipped
              with a warning and the full count is used.

              See :ref:`Covering All Combinations <covering-all-combinations>`
              for more information.

              :param factors: the factors whose level combinations must all appear
              :type factors: Factor
              :param prioritize: the factors to keep fully crossed, or a
                                 boolean selecting full coverage or an
                                 interactive choice
              :type prioritize: Union[bool, List[Factor]]
              :rtype: Constraint

.. class:: sweetpea.ContinuousConstraint(factors, predicate)

              Constrains :class:`.ContinuousFactor` in an experiment so that 
              the samples generated for these factors meet the proposed 
              constraint function. Since such constraints only apply to 
              factors with continuous sampling functions that 
              should not be included in the crossing, the experiment will 
              sample these factors until the constraints are met after 
              the trial sequences have been satified for discrete factors.

              :param factors: the factors to add constraints on
              :type factors: List[ContinuousFactor]
              :param predicate: a constraint function takes `factors`
                                initialized with sampling function.
                                The function should return true if the
                                combination of factors meet the constraints.
              :type predicate: Callable[[Any, ...], bool]
              :rtype: Constraint
