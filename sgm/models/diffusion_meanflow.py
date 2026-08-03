"""Teacher-free training engine for MeanFlow cloud removal."""

from .diffusion import ResidualDiffusionEngine


class MeanFlowEngine(ResidualDiffusionEngine):
    """Run MeanFlow with the standard model EMA and residual evaluation stack.

    MeanFlow builds its derivative target from the student itself, so it does
    not need the separate frozen EMA teacher used by CFM.  The ordinary
    ``MeanFlowLoss`` keeps a teacher argument only for call-signature
    compatibility; this engine passes ``None`` in that position.
    """

    def __init__(self, *args, **kwargs):
        # Validation, test, prediction and image logging use ema_scope() from
        # ResidualDiffusionEngine, so keep the regular model EMA enabled.
        kwargs["use_ema"] = True
        super().__init__(*args, **kwargs)
        self._checked_training_gradients = False

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Accept checkpoints written by the former CFM-backed configuration.

        Those checkpoints contain a frozen ``teacher_model`` subtree that has
        no counterpart in this teacher-free engine.  All student, conditioner
        and regular EMA entries retain their original names.
        """
        teacher_keys = [
            key for key in state_dict if key.startswith("teacher_model.")
        ]
        if teacher_keys:
            original_state_dict = state_dict
            state_dict = state_dict.copy()
            if hasattr(original_state_dict, "_metadata"):
                state_dict._metadata = original_state_dict._metadata
            for key in teacher_keys:
                del state_dict[key]
        return super().load_state_dict(
            state_dict,
            strict=strict,
            assign=assign,
        )

    def on_after_backward(self):
        """Report parameters skipped by the first MeanFlow backward graph."""
        super().on_after_backward()
        if self._checked_training_gradients:
            return

        unused = [
            name
            for name, param in self.model.named_parameters()
            if param.requires_grad and param.grad is None
        ]
        if unused and getattr(self, "global_rank", 0) == 0:
            preview = "\n  - ".join(unused[:50])
            suffix = (
                f"\n  ... and {len(unused) - 50} more"
                if len(unused) > 50
                else ""
            )
            print(
                "[MeanFlow/DDP] Parameters without gradients in the first "
                "backward pass:\n  - "
                f"{preview}{suffix}"
            )

        self._checked_training_gradients = True

    def forward(self, x, mu, batch):
        """Evaluate the teacher-free MeanFlow loss on a clean/cloudy pair."""
        loss = self.loss_fn(
            self.model,
            None,
            self.denoiser,
            self.conditioner,
            self.sigma2st,
            x,
            mu,
            batch,
        )
        loss_mean = loss.mean()
        return loss_mean, {"loss": loss_mean}
