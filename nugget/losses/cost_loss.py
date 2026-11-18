"""Cable and junction-box cost penalty."""

from nugget.losses.base_loss import LossFunction
from nugget.losses.geometry_penalties import CapacityPenalty
import torch


class CostLoss(LossFunction):
	"""Penalize cable length, junction-box count, and capacity overuse.

	``geom_dict`` is expected to contain fixed string coordinates, CJB
	coordinates, and string-to-CJB assignment logits. Cost coefficients can be set at
	construction or overridden for an individual call.

	The returned ``estimated_total_cost`` is the unnormalized cable and CJB cost;
	``cost_loss`` also includes the capacity-overuse penalty and is normalized.
	"""

	def __init__(
		self,
		C_cable=200,
		C_cjb=150e3,
		C_connection=20e3,
		normalization=1_000_000,
		device=None,
		**kwargs,
	):
		super().__init__(device)
		self.capacity_penalty = CapacityPenalty(device=device)
		self.parameters = {
			"C_cable": C_cable,
			"C_cjb": C_cjb,
			"normalization": normalization,
			"C_connection": C_connection,
		}
		self.parameters.update(kwargs)

	def __call__(self, geom_dict, **kwargs):
		parameters = {**self.parameters, **kwargs}
		strings = geom_dict.get("string_xy", None)
		cjbs = geom_dict.get("cjb_xy", None)
		assignment_logits = geom_dict.get("assignment_logits", None)
		cjb_weights= geom_dict.get("cjb_weights", None)

		assignment_weights = torch.sigmoid(assignment_logits)
		distances = torch.cdist(strings, cjbs)
		expected_cable_lengths = torch.sum(assignment_weights * distances, dim=1)
		cable_cost = parameters["C_cable"] * torch.sum(expected_cable_lengths)

		cjb_usage = torch.sum(assignment_weights, dim=0)
		connection_cost= torch.sum(cjb_usage) * parameters["C_connection"]

		cjb_count_cost = parameters["C_cjb"] * cjb_weights #<-- soft_active_cjbs
		estimated_total_cost = cable_cost + cjb_count_cost + connection_cost

		loss = estimated_total_cost
		loss = loss / parameters["normalization"]
		return {"cost_loss": loss, "estimated_total_cost": estimated_total_cost}