"""Software assumptions for puck pickup, transport, and dropoff."""

from dataclasses import dataclass

from .models import PuckState


@dataclass(frozen=True)
class PuckTransportAssumptions:
    passive_carrier: bool = True
    requires_operator_load: bool = True
    requires_operator_unload: bool = True


def require_puck_loaded(puck_state: PuckState) -> None:
    if puck_state != PuckState.HELD:
        raise ValueError("puck must be loaded before transport")

