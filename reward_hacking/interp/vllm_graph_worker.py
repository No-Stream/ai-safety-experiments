"""vLLM V1 worker that prepares residual buffers before model profiling and capture."""

from __future__ import annotations

from vllm.v1.worker.gpu_worker import Worker

from reward_hacking.interp.vllm_graph_buffers import validate_graph_compiler
from reward_hacking.interp.vllm_interventions import prepare_graph_interventions


class ResidualGraphWorker(Worker):
    """Use the standard V1 lifecycle with a pre-trace residual-edit installation."""

    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        """Load normally, then install before determine_available_memory's first forward."""
        validate_graph_compiler()
        super().load_model(load_dummy_weights=load_dummy_weights)
        additional_config = self.vllm_config.additional_config
        if not isinstance(additional_config, dict):
            raise TypeError("graph residual worker requires dictionary additional_config")
        settings = additional_config["residual_intervention_graph"]
        layer_ranks = {int(layer): int(rank) for layer, rank in settings["layer_ranks"].items()}
        prepare_graph_interventions(self.get_model(), layer_ranks)
