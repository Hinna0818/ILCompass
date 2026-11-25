"""Shared outer fitting and A/B prediction boundaries for the mouse dictionary."""

import numpy as np
import pandas as pd


def outer_fit(rows, fold):
    return rows[(rows.processing_batch != fold.batch) & (rows.cell_group != fold.hold_cell_group)
                & (rows.ligand_group != fold.hold_ligand_group)]


def routed_rows(rows, fit, fold):
    test = rows[rows.processing_batch == fold.batch]
    task_a = test[(test.cell_group == fold.hold_cell_group)
                  & (test.ligand_group % 3 + 1 == fold.hold_ligand_group)
                  & test.split_ligand.isin(fit.split_ligand)]
    task_b = test[(test.cell_group % 3 + 1 == fold.hold_cell_group)
                  & (test.ligand_group == fold.hold_ligand_group)
                  & test.celltype.isin(fit.celltype)]
    routed = pd.concat([task_a.assign(task="A"), task_b.assign(task="B")], ignore_index=True)
    routed.insert(0, "prediction_index", np.arange(len(routed)))
    return routed
