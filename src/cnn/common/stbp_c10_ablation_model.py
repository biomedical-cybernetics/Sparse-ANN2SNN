"""Parent graph and mask semantics with a bounded Linear evolution clock."""
from stbp_budget_model import make_model as parent_model, BudgetLinearScheduler


def make_model(budget, dropout):
    model = parent_model(budget)
    model.classifier[3].p = dropout
    return model


class FrozenBudgetLinearScheduler(BudgetLinearScheduler):
    def __call__(self):
        # Optimizer wrappers and backward hooks still enforce the fixed mask.
        if self.step >= self.T_end:
            return True
        return super().__call__()
