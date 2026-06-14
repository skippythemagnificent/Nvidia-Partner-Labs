"""Prod networking (VPC, subnets, security groups) — scaffold only."""
from __future__ import annotations

import pulumi


class Networking(pulumi.ComponentResource):
    def __init__(self, name: str, cfg, opts=None) -> None:
        super().__init__("nvidia-labs:infra:Networking", name, {}, opts)
        # TODO prod: VPC, public+private subnets, NAT, GPU node SGs.
        self.register_outputs({})
