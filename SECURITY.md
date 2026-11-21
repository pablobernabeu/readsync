# Security and data protection

readsync records data from human participants. This document sets out what the toolkit protects, how it does so and where that protection ends.

## What is protected

The protections below apply to the session log that `EventLog` writes. It holds the gaze samples polled during each passage, the passage, word and region events, the comprehension responses and the calibration and drift-check outcomes. A session also produces files that readsync does not encrypt: the event and quality exports, which are plain CSV and JSON, and the EyeLink data file (EDF), which holds the full-rate gaze record. These need the same storage controls as any participant data. The interest-area files hold stimulus geometry only.

## Pseudonymisation

`pseudonymise()` derives a 16-character code from an identifier with HMAC-SHA256 under a study key. The same identifier and key always give the same code, so a participant's sessions can be linked, and the identifier cannot be recovered from the code without the key. The identifier is put in Unicode NFC form and stripped of surrounding whitespace first, but its case is kept, so identifiers that should match whatever their case, such as email addresses, need to be lower-cased beforehand. `ReadingSession` records whatever participant string it receives, so pass it the code and never the identifier. Keep the pseudonymisation key separate from the encryption key. `new_data_key()` generates either.

## Encryption at rest

Each record in the log is encrypted with [Fernet](https://github.com/fernet/spec/blob/master/Spec.md), which uses AES-128 in CBC mode and authenticates the whole token with HMAC-SHA256. Without the key, the contents of a record cannot be read, and any change to a record is detected when it is decrypted. The file does not hide how many records it holds, how large each one is or when each was written, because every Fernet token carries its creation time in clear.

## Tamper evidence

Each record carries a sequence number and the hash of the record before it, so editing, deleting or reordering records inside the log is detected when the log is read, and the export and quality functions refuse a log that fails the check. Removing records from the end leaves a valid shorter chain, but a truncated log can still be recognised. A session that finishes normally ends with a `session_end` event whose `status` is `completed`, its `session_start` event lists the passages it planned, and the quality report marks any passage that has no offset. For stronger assurance, store the log's head, which `EventLog.head` and the session result report as the record count and the last chain hash, separately from the log. A log whose last write was cut short by a crash no longer verifies, and `recover_events` returns every record before the damage.

## Offline guard

`ReadingSession` runs inside a `NetworkGuard` by default. The guard replaces `connect` and `connect_ex` on Python's `socket.socket` class for the duration of the session, so an outbound connection attempted by Python code, such as an upload by a dependency or a stray analytics call, raises `OfflineViolation`. Connections to loopback addresses and Unix-domain sockets stay open, so local Python services keep working. A proxy listening on a loopback address, for example one named in `HTTPS_PROXY`, would therefore let requests leave through it. On a machine that runs such a proxy, call the session's `run` inside `with NetworkGuard(allow_loopback=False):`. While several guards are active, a connection must be allowed by all of them.

The guard does not stop a determined adversary, and it does not cover every route to a network. DNS lookups still reach the resolver before a blocked connection is refused. Connectionless sends such as UDP `sendto`, connections opened before the guard started, inbound connections, child processes and code that calls the `_socket` module directly are not intercepted. Native libraries open their own sockets, so the EyeLink link and Lab Streaming Layer keep working under the guard, and the marker stream remains visible on the local network. For stronger isolation, run sessions on a machine disconnected from external networks.

## Limits

readsync defends against accidental disclosure and casual tampering: an upload triggered by a dependency, a hand-edited or corrupted data file or a stolen disk holding encrypted logs. Whoever holds the keys can read and rewrite the data, so store the keys apart from the data and under stricter access control, for example in the institution's secrets store. The toolkit does not by itself make a study compliant with data-protection law. Consent, lawful basis, retention and access remain the researcher's responsibility under the institution's approvals and the applicable data-protection law, such as the UK GDPR.

## Dependencies

The core depends only on `cryptography`, which provides the encryption and the hashing. PsychoPy, pylink, pylsl, MediaPipe and wordfreq are optional extras, imported only by the modules that use them, so none of them is loaded when a log is written, read or exported unless the session itself uses it.

## Reporting a vulnerability

Report suspected vulnerabilities through GitHub's private vulnerability reporting on this repository, and allow reasonable time for a fix before public disclosure.
