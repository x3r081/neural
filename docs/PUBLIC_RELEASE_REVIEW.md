# Public release review â€” October 3, 2026

The public `x3r081/neural` repository starts from the current validated runtime with a clean Git history. Earlier development history is retained privately: publishing only a cleanup commit would still expose personal information in older commits.

## What was checked

- Gitleaks 8.30.1 scanned both pre-publication origin branches and all 57 reachable commits, plus a clean snapshot of the 638 tracked source files. Its official download checksum was verified.
- All 127 history detections were reviewed. They were benchmark token-ID hashes, a model identifier, or tokenizer metadata; no actual credentials were identified.
- The content review checked benchmark requests, outputs, logs, and commit identities for personal information. Old commit identities remain in the private archive. Personal Windows profile labels in 81 current benchmark files and documents were replaced with `USER`.
- The public commit uses the repository owner's GitHub noreply identity. Local configuration, credentials, models, environments, caches, and private audit files are excluded from the published source.

## Benchmark provenance

Only personal path labels were redacted from the historical log and documentation. Performance numbers, model outputs, model parameters, and validated kernel/report bytes were not changed. References to older source commit IDs describe private development provenance; public benchmark files are included in this repository.

The release does not claim a new performance gain or a new full-model benchmark. The guided setup and launcher were tested in the preceding release. Model weights are downloaded separately from their official, revision-pinned source by the installer.
