"""NIM cluster component — deploys NIM Helm charts on a K8s target.

Scaffold: only the LLM NIM is wired up. Embed/rerank releases are added in
Lab 03 when learners walk through multi-NIM topology.
"""
from __future__ import annotations

import base64
import json

import pulumi
import pulumi_kubernetes as k8s
from pulumi_kubernetes.helm.v3 import Release, ReleaseArgs, RepositoryOptsArgs


def _docker_config(server: str, username: str, password: str) -> str:
    auth = base64.b64encode(f"{username}:{password}".encode()).decode()
    return json.dumps({"auths": {server: {"auth": auth}}})


class NimCluster(pulumi.ComponentResource):
    def __init__(self, name: str, cfg, vector_store, opts=None) -> None:
        super().__init__("nvidia-labs:infra:NimCluster", name, {}, opts)

        provider = k8s.Provider(
            f"{name}-provider",
            kubeconfig=cfg.kubeconfig,
            opts=pulumi.ResourceOptions(parent=self),
        )

        ns = k8s.core.v1.Namespace(
            f"{name}-ns",
            metadata={"name": cfg.namespace},
            opts=pulumi.ResourceOptions(parent=self, provider=provider),
        )

        ngc_secret = k8s.core.v1.Secret(
            f"{name}-ngc-secret",
            metadata={"name": "ngc-secret", "namespace": cfg.namespace},
            string_data={
                ".dockerconfigjson": cfg.ngc_api_key.apply(
                    lambda key: _docker_config("nvcr.io", "$oauthtoken", key)
                )
            },
            type="kubernetes.io/dockerconfigjson",
            opts=pulumi.ResourceOptions(parent=self, provider=provider, depends_on=[ns]),
        )

        self._llm_release = Release(
            f"{name}-llm",
            ReleaseArgs(
                chart="nim-llm",
                repository_opts=RepositoryOptsArgs(
                    repo="https://helm.ngc.nvidia.com/nim/charts",
                    username="$oauthtoken",
                    password=cfg.ngc_api_key,
                ),
                namespace=cfg.namespace,
                values={
                    "image": {"repository": f"nvcr.io/nim/{cfg.nim_llm_model}"},
                    "model": {"ngcAPIKey": cfg.ngc_api_key},
                    "resources": {"limits": {"nvidia.com/gpu": cfg.gpu_count}},
                    "persistence": {"enabled": True, "size": "50Gi"},
                },
            ),
            opts=pulumi.ResourceOptions(parent=self, provider=provider, depends_on=[ngc_secret]),
        )

        self.embed_url = pulumi.Output.concat(
            "http://nim-embed.", cfg.namespace, ".svc.cluster.local:8080/v1"
        )
        self.rerank_url = pulumi.Output.concat(
            "http://nim-rerank.", cfg.namespace, ".svc.cluster.local:8082/v1"
        )
        self.llm_url = pulumi.Output.concat(
            "http://nim-llm.", cfg.namespace, ".svc.cluster.local:8000/v1"
        )

        self.register_outputs(
            {"embed_url": self.embed_url, "rerank_url": self.rerank_url, "llm_url": self.llm_url}
        )
