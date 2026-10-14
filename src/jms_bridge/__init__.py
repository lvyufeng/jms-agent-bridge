"""jms-agent-bridge — an MCP bridge from Claude Code to a JumpServer bastion.

Control plane (asset discovery, audit, tickets) goes over JumpServer's REST API;
data plane (commands, file sync, jobs) goes over SSH through the KoKo gateway.
"""

__version__ = "0.2.2"