"""Cut-point phase — is there a threshold, where is it, and can we defend it?

Built step by step in ``meningioma-cutpoints.ipynb``. One module per question,
in the order the notebook asks them:

``cohort``      what have we got, and is it one row per patient?

Nothing here writes back into the cleaning handoff. A cut-point estimated from
this cohort has already seen the outcome, so feeding it into the imputation or
the multivariable models would leak the answer into the predictors.
"""
