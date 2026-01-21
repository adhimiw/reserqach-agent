# Biometric login (simulation)

SupportX includes a **biometric login simulation** for demo purposes.

- The server issues a short-lived **challenge**.
- The client signs the challenge using a locally stored **biometric token**.
- The server verifies the signature and returns a session token.

No real biometric data (fingerprints/face scans) is stored.
