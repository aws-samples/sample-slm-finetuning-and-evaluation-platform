# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Backend application package.

Importing anything from this package first makes every AWS SDK call made in the
process identify this solution in its user-agent, which is how the solution's
API usage is measured. It has to happen here rather than in each entrypoint:
this module runs before any submodule, so no client can be constructed ahead of
it. It is a no-op when no user-agent is configured.
"""
from .aws_clients import install as _install_solution_user_agent

_install_solution_user_agent()
