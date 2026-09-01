# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""HF token guard logic (no real Secrets Manager calls)."""
import app.secrets as secrets


def test_no_secret_name_means_no_token(monkeypatch):
    monkeypatch.delenv("SLM_HF_SECRET_NAME", raising=False)
    assert secrets.get_hf_token() is None
    assert secrets.hf_token_is_set() is False


def test_hf_prefix_required(monkeypatch):
    monkeypatch.setenv("SLM_HF_SECRET_NAME", "x")
    # Stub the secrets-manager fetch to return various values.
    def fake(value):
        monkeypatch.setattr(
            secrets, "_client",
            lambda: type("C", (), {"get_secret_value": lambda self, **k: {"SecretString": value}})(),
        )

    fake("UNSET")
    assert secrets.get_hf_token() is None  # placeholder rejected
    fake("garbage")
    assert secrets.get_hf_token() is None  # non-hf rejected
    fake("hf_realtoken123")
    assert secrets.get_hf_token() == "hf_realtoken123"
    assert secrets.hf_token_is_set() is True


def test_fetch_error_means_no_token(monkeypatch):
    monkeypatch.setenv("SLM_HF_SECRET_NAME", "x")

    def boom():
        raise RuntimeError("no access")

    monkeypatch.setattr(secrets, "_client", boom)
    assert secrets.get_hf_token() is None
