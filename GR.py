import torch
import torch.nn.functional as F


class GradientRebalancer:
    """GR algorithm core for modality-aware recommendation gradients.

    This is intentionally a plain Python object rather than ``nn.Module``:
    GR owns transient control state but no trainable parameters, matching the
    previous checkpoint behavior of R2ARec.
    """

    def __init__(self, enabled, neg_num, tau, epsilon, warmup_steps):
        self.enabled = bool(enabled)
        self.neg_num = int(neg_num)
        self.tau = float(tau)
        self.epsilon = float(epsilon)
        self.warmup_steps = int(warmup_steps)

        if self.neg_num < 1:
            raise ValueError('gr_neg_num must be at least 1.')
        if self.tau <= 0:
            raise ValueError('gr_tau must be positive.')
        if self.epsilon <= 0:
            raise ValueError('gr_epsilon must be positive.')
        if self.warmup_steps < 0:
            raise ValueError('gr_warmup_steps cannot be negative.')

        self.kappas = None
        self.previous_kappas = None
        self.last_coefficients = {'v': 1.0, 't': 1.0}
        self.step = 0

    @torch.no_grad()
    def compute_kappas(
        self,
        users,
        pos_items,
        gr_neg_items,
        fused_users,
        fused_items,
        modal_views,
        n_users,
        n_items,
    ):
        """Return detached KL controls on a shared positive-plus-negative support."""
        support = torch.cat((pos_items.unsqueeze(0), gr_neg_items), dim=0).transpose(0, 1)
        fused_scores = (fused_users.unsqueeze(1) * fused_items[support]).sum(dim=-1)
        teacher_probs = F.softmax(fused_scores / self.tau, dim=-1).detach()
        teacher_log = teacher_probs.clamp_min(self.epsilon).log()

        kappas = {}
        for name in ('v', 't'):
            modal_users, modal_items = torch.split(
                modal_views[name], [n_users, n_items], dim=0
            )
            modal_scores = (
                modal_users[users].unsqueeze(1) * modal_items[support]
            ).sum(dim=-1)
            modal_probs = F.softmax(modal_scores / self.tau, dim=-1)
            modal_log = modal_probs.clamp_min(self.epsilon).log()
            kappas[name] = (
                modal_probs * (modal_log - teacher_log)
            ).sum(dim=-1).mean().detach()
        return kappas

    def update_kappas(self, *args, **kwargs):
        self.kappas = self.compute_kappas(*args, **kwargs)
        return self.kappas

    def clear_kappas(self):
        self.kappas = None

    def coefficients(self):
        """Compute Eq. (11)-(12) coefficients from one-step KL progress."""
        ones = {'v': 1.0, 't': 1.0}
        if not self.enabled or self.kappas is None:
            return ones

        if self.previous_kappas is None:
            deltas = {
                name: value.clamp_min(0.0)
                for name, value in self.kappas.items()
            }
        else:
            deltas = {
                name: (self.previous_kappas[name] - value).clamp_min(0.0)
                for name, value in self.kappas.items()
            }

        self.previous_kappas = {
            name: value.detach().clone() for name, value in self.kappas.items()
        }
        current_step = self.step
        self.step += 1

        if current_step < self.warmup_steps:
            return ones
        if not all(torch.isfinite(value).item() for value in deltas.values()):
            return ones

        total_delta = sum(deltas.values())
        if total_delta.item() == 0.0:
            return ones

        modality_count = len(deltas)
        denominator = total_delta + modality_count * self.epsilon
        scale = modality_count / (modality_count - 1)
        return {
            name: (
                scale
                * (total_delta - delta + self.epsilon)
                / denominator
            ).item()
            for name, delta in deltas.items()
        }

    def backward(
        self,
        recommendation_loss,
        contrastive_loss,
        parameters,
        modal_encoder_params,
    ):
        """Scale only modality-encoder recommendation gradients and add Lcon gradients."""
        parameters = [parameter for parameter in parameters if parameter.requires_grad]
        rec_grads = torch.autograd.grad(
            recommendation_loss, parameters, retain_graph=True, allow_unused=True
        )
        con_grads = torch.autograd.grad(
            contrastive_loss, parameters, allow_unused=True
        )

        coefficients = self.coefficients()
        self.last_coefficients = coefficients.copy()
        modality_by_parameter = {
            id(parameter): name
            for name, modal_parameters in modal_encoder_params.items()
            for parameter in modal_parameters
        }
        for parameter, rec_grad, con_grad in zip(parameters, rec_grads, con_grads):
            gradient = None
            if rec_grad is not None:
                modality = modality_by_parameter.get(id(parameter))
                gradient = (
                    rec_grad
                    if modality is None
                    else rec_grad * coefficients[modality]
                )
            if con_grad is not None:
                gradient = con_grad if gradient is None else gradient + con_grad
            parameter.grad = gradient

        return coefficients
