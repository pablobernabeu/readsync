# Security and data protection

`readsync` records data from human participants, so data protection is part of the design. This document states what is guaranteed, how, and where the limits are.

## What is protected

Reading-session records: gaze samples, region and passage events, responses and markers. These are linked to a participant by a pseudonym, never by a direct identifier.

## Guarantees

1. **Pseudonymisation at source.** `pseudonymise()` derives a stable code from an identifier with a keyed HMAC (HMAC-SHA256). The same person maps to the same code within a study, so sessions can be linked, while the original identifier cannot be recovered from the code without the secret study key. Direct identifiers are never written to the data files.

2. **Encryption at rest.** Every record in the event log is encrypted with an authenticated cipher (Fernet, AES-128 in CBC mode with an HMAC). The file is opaque on disk, and any edit to a record is detected on read.

3. **Tamper-evidence.** Records form a hash chain. Each record carries the hash of the previous one, so edits, deletion or reordering of interior records are detected. A failed check stops an export. Removing the newest records leaves a valid shorter chain, so completeness is judged against the closing `session_end` event that every completed session appends; a study needing stronger assurance records the final chain hash out of band.

4. **Offline during a session.** `ReadingSession` runs inside a `NetworkGuard` by default, which blocks outbound network connections for the duration of the recording. Participant data cannot be uploaded during a session, and experiment timing is not disturbed by network activity. Loopback is allowed so that local EEG marker transport over Lab Streaming Layer keeps working.

## Threat model and limits

`readsync` defends against accidental disclosure and casual tampering: an upload triggered by a dependency, a stray analytics call, a hand-edited or corrupted data file, or a stolen disk where the data are encrypted. It does not aim to contain a determined adversary.

- The encryption key and the pseudonymisation key are held by the researcher. Store them separately from the data, under stricter access control, for example in the institution's secrets store. Whoever holds the key can read the data; that is the intended design.
- `NetworkGuard` intercepts socket `connect` and `connect_ex` calls made from Python code. Connectionless sends such as UDP `sendto`, connections established before the guard started, and native-code transports that bypass Python's socket layer are not intercepted, and it does not stop a determined adversary with code execution on the machine. For stronger isolation, run sessions on a machine with networking disabled.
- The toolkit does not, by itself, make a study compliant with data-protection law. It provides controls that support compliance. Consent, lawful basis, retention and access remain the researcher's responsibility under the institution's approvals and UK GDPR.

## Dependencies

The core depends only on `cryptography`, a widely used, audited library. Heavy and hardware-specific dependencies are optional extras, which keeps the trusted base small. Dependencies are constrained with lower bounds and reviewed before updates.

## Reporting a vulnerability

Report suspected vulnerabilities through GitHub's private vulnerability reporting on the repository, and allow reasonable time for a fix before public disclosure.
