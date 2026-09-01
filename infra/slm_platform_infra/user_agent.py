# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Hand the solution's user-agent suffix to the compute that makes AWS calls.

API usage is attributed to this solution by a suffix on the user-agent of every
AWS SDK call it makes. The value is rendered once (solution_config) and reaches
the code that needs it as the USER_AGENT_STRING environment variable.

For Lambda, an Aspect is the right tool: it visits every node of the construct
tree at synth time, so a function added later is covered without anyone
remembering to wire it. The catch is that an Aspect only covers what it matches
— any compute that is not an `aws_lambda.Function` is structurally out of reach
and has to take the value through its own constructor or deploy-time
configuration. In this app that means:

  * the three container Lambdas (api, reconcile, worker) — covered here;
  * CDK's own helper Lambdas (bucket deployment, auto-delete, custom resources)
    — also matched, but they run CDK's JavaScript, which never reads the
    variable, so it is inert there;
  * the CodeBuild project that builds the training image — makes its AWS calls
    through the AWS CLI, whose user-agent cannot carry this suffix verbatim
    (the only knob available sanitises the value), so it is not attributed;
  * the agent runtime — deployed outside this app, so it takes the value from
    its own deploy-time environment.
"""
from __future__ import annotations

import aws_cdk as cdk
import jsii
from aws_cdk import aws_lambda as lambda_
from constructs import IConstruct

#: Name of the environment variable the runtime code reads. Consumers use
#: os.environ.get with a default, so setting it is never load-bearing: the value
#: only ever adds a suffix to a user-agent.
ENV_VAR_NAME = "USER_AGENT_STRING"


@jsii.implements(cdk.IAspect)
class SolutionUserAgentAspect:
    """Set USER_AGENT_STRING on every Lambda function in the visited scope."""

    def __init__(self, user_agent: str) -> None:
        self._user_agent = user_agent

    def visit(self, node: IConstruct) -> None:
        if not self._user_agent:
            # No identity configured (a fork may have blanked it) — leave the
            # environment untouched rather than setting an empty variable.
            return
        if isinstance(node, lambda_.Function):
            node.add_environment(ENV_VAR_NAME, self._user_agent)
