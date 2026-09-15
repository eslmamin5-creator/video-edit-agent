"""Order resolution (spec section 5) -- non-negotiable: the AI must never
silently reorder scenes. Discovery already yields a deterministic natural
filename order (see `discovery.py`); this module decides which order is
actually used for the timeline, and always records the policy that was
applied so the decision is auditable.

- PRESERVE (default): use the order scenes were discovered/supplied in.
- FILENAME: explicit request for deterministic natural filename order
  (identical to PRESERVE when scenes came from directory discovery, since
  discovery already sorts naturally; kept distinct for explicit CLI args
  where the caller supplied an arbitrary list).
- SCRIPT: use the script-alignment recommendation, but only when the
  caller explicitly opted in via `--order script`; a low-confidence
  alignment falls back to the discovered order with a warning rather than
  applying a guess.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from video_edit_agent.agents.assembler.schemas import (
    OrderPolicy,
    SceneInventoryItem,
    ScriptAlignmentItem,
)

MIN_SCRIPT_ORDER_CONFIDENCE = 0.5


@dataclass
class OrderResolution:
    ordered_items: list[SceneInventoryItem]
    policy_applied: OrderPolicy
    recommendation: list[str] | None = None  # scene ids in the AI's recommended order, if different
    warnings: list[str] = field(default_factory=list)


def resolve_order(
    items: list[SceneInventoryItem],
    *,
    policy: OrderPolicy = OrderPolicy.PRESERVE,
    script_alignment: list[ScriptAlignmentItem] | None = None,
) -> OrderResolution:
    discovered_order = list(items)  # already naturally sorted by discovery
    warnings: list[str] = []
    recommendation: list[str] | None = None

    if script_alignment:
        aligned = [a for a in script_alignment if a.script_beat_index is not None]
        if aligned:
            avg_conf = sum(a.confidence for a in aligned) / len(aligned)
            if avg_conf >= MIN_SCRIPT_ORDER_CONFIDENCE and len(aligned) == len(items):
                by_id = {it.id: it for it in items}
                rec_order = sorted(aligned, key=lambda a: a.script_beat_index)
                recommendation = [a.scene_id for a in rec_order]

    if policy == OrderPolicy.PRESERVE:
        return OrderResolution(ordered_items=discovered_order, policy_applied=policy, recommendation=recommendation)

    if policy == OrderPolicy.FILENAME:
        # Discovery already applies natural filename order.
        return OrderResolution(ordered_items=discovered_order, policy_applied=policy, recommendation=recommendation)

    if policy == OrderPolicy.SCRIPT:
        if recommendation is None:
            warnings.append(
                "--order script requested but script alignment confidence was insufficient; "
                "falling back to discovered (filename) order."
            )
            return OrderResolution(
                ordered_items=discovered_order, policy_applied=OrderPolicy.PRESERVE, warnings=warnings
            )
        by_id = {it.id: it for it in items}
        ordered = [by_id[sid] for sid in recommendation]
        return OrderResolution(ordered_items=ordered, policy_applied=policy, recommendation=recommendation)

    raise ValueError(f"Unknown order policy: {policy}")
