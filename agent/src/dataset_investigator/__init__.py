# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Agentic dataset-investigation agent (Strands + Bedrock AgentCore)."""
# Before anything that might build an AWS client — including Strands, which
# builds its own Bedrock client — make this process's AWS calls identify the
# solution, which is how its API usage is measured. A no-op when the runtime was
# launched without USER_AGENT_STRING set.
from .aws_user_agent import install as _install_solution_user_agent

_install_solution_user_agent()

from .core import generate_questions, synthesize_proposal  # noqa: E402

__all__ = ["generate_questions", "synthesize_proposal"]
