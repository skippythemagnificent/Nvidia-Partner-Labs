"""kube-prometheus-stack + DCGM exporter — used by Lab 06."""
from __future__ import annotations

import pulumi
from pulumi_kubernetes.helm.v3 import Release, ReleaseArgs, RepositoryOptsArgs


class MonitoringStack(pulumi.ComponentResource):
    def __init__(self, name: str, cfg, nim_cluster, opts=None) -> None:
        super().__init__("nvidia-labs:infra:MonitoringStack", name, {}, opts)

        Release(
            f"{name}-kube-prometheus",
            ReleaseArgs(
                chart="kube-prometheus-stack",
                repository_opts=RepositoryOptsArgs(
                    repo="https://prometheus-community.github.io/helm-charts"
                ),
                namespace=cfg.namespace,
                values={
                    "grafana": {"enabled": True, "adminPassword": "nvidia-labs"},
                    "prometheus": {
                        "prometheusSpec": {
                            "additionalScrapeConfigs": [
                                {
                                    "job_name": "triton-nim",
                                    "static_configs": [{"targets": ["nim-llm:8002"]}],
                                    "metrics_path": "/metrics",
                                }
                            ]
                        }
                    },
                },
            ),
            opts=pulumi.ResourceOptions(parent=self),
        )

        Release(
            f"{name}-dcgm",
            ReleaseArgs(
                chart="dcgm-exporter",
                repository_opts=RepositoryOptsArgs(
                    repo="https://nvidia.github.io/dcgm-exporter/helm-charts"
                ),
                namespace=cfg.namespace,
                values={"serviceMonitor": {"enabled": True}},
            ),
            opts=pulumi.ResourceOptions(parent=self),
        )

        self.prometheus_url = (
            f"http://kube-prometheus-stack-prometheus.{cfg.namespace}.svc:9090"
        )
        self.grafana_url = f"http://kube-prometheus-stack-grafana.{cfg.namespace}.svc"

        self.register_outputs(
            {"prometheus_url": self.prometheus_url, "grafana_url": self.grafana_url}
        )
