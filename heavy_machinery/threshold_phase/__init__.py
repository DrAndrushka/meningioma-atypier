"""Threshold phase — cut-points, risk curves, and multi-cut rules.

Consumes the same ``output/datasets/`` handoff as the modelling notebook and
answers three questions that the multivariable models do not:

``thresholds``    which single cut-point separates the grades best, and how
                  stable is that number?
``risk_curves``   where does the *risk* of high grade climb most steeply — the
                  dose-response question, which a ROC cannot answer.
``combinations``  does using two or more cut-points together beat the best
                  single one?

``stability`` re-runs all three across the m MICE draws; ``artifacts`` writes
figures and tables to ``output/thresholds/``.

Nothing here writes back into the cleaning handoff. Cut-points estimated from
this cohort have already seen the outcome, so feeding them into the imputation
or the multivariable models would leak the answer into the predictors.
"""
