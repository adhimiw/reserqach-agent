# Dead Man’s Switch

SupportX runs a Dead Man’s Switch that expects periodic heartbeats.

If the service stops sending heartbeats for longer than the configured timeout:
- An alert event is triggered.
- Optionally, a webhook is called (see `SUPPORTX_DMS_WEBHOOK_URL`).

This is designed to demonstrate safety/monitoring for unattended agents.
