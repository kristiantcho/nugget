"""Cable and junction-box cost penalty."""

from nugget.losses.base_loss import LossFunction
import torch
import torch.nn.functional as F


class CostLoss(LossFunction):
	"""Penalize cable length, junction-box count, and capacity overuse.

	``geom_dict`` is expected to contain fixed string coordinates, CJB
	coordinates, and string-to-CJB assignment logits. Cost coefficients can be set at
	construction or overridden for an individual call.
	"""

	def __init__(
		self,
		C_cable=200,
		C_cjb=150e3,
		cjb_capacity=10,
		C_capacity_penalty=50_000_000,
		normalization=1_000_000,
		capacity_beta=5,
		device=None,
		**kwargs,
	):
		super().__init__(device)
		self.parameters = {
			"C_cable": C_cable,
			"C_cjb": C_cjb,
			"cjb_capacity": cjb_capacity,
			"C_capacity_penalty": C_capacity_penalty,
			"normalization": normalization,
			"capacity_beta": capacity_beta,
		}
		self.parameters.update(kwargs)

	def __call__(self, geom_dict, **kwargs):
		parameters = {**self.parameters, **kwargs}
		strings = geom_dict.get("string_xy", None)
		cjbs = geom_dict.get("cjb_xy", None)
		assignment_logits = geom_dict.get("assignment_logits", None)

		strings = torch.as_tensor(strings).detach()
		cjbs = torch.as_tensor(cjbs, dtype=strings.dtype, device=strings.device)
		assignment_logits = torch.as_tensor(
			assignment_logits, dtype=strings.dtype, device=strings.device
		)
		assignment_weights = assignment_logits

		distances = torch.cdist(strings, cjbs)
		expected_cable_lengths = torch.sum(assignment_weights * distances, dim=1)
		cable_cost = parameters["C_cable"] * torch.sum(expected_cable_lengths)

		cjb_usage = torch.sum(assignment_weights, dim=0)
		cjb_count_cost = parameters["C_cjb"] * torch.sum(torch.sigmoid(cjb_usage*6))
		excess_usage = F.softplus(
			cjb_usage - parameters["cjb_capacity"],
			beta=parameters["capacity_beta"],
		)
		capacity_cost = parameters["C_capacity_penalty"] * torch.sum(
			excess_usage.square()
		)

		loss = (cable_cost + cjb_count_cost + capacity_cost) / parameters[
			"normalization"
		]
		return {"cost_loss": loss}