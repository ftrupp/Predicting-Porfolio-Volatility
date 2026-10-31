# Release hygiene and licensing

The current experiment has no account credentials, broker integration, download
fallback or network call. The local adapter accepts CSV and provenance JSON.
Legacy pickles, reports, checkpoints, bytecode and import-time download helpers
are excluded from the current release tree. Existing local copies may remain
ignored for their owner's use; they are not part of the package or examples.

A FRED credential literal was present in the previously published notebook and
its checkpoint at commit `af3f3d12ce50d3857f0826d0685252acecef5725` (August 8, 2025).
This was confirmed through read-only public-file inspection on October 30, 2026.
The value was not tested or reproduced in the new artifacts. Removing it from
the current files does not revoke it or remove earlier Git history. Its owner
should replace it if it remains active. This release does not rewrite history.

`python scripts/check_release.py` scans the staged Git tree and prints paths and
rule names rather than matched sensitive values. `--revision COMMIT` checks an
immutable tree. It catches credential literals, common tokens, private paths,
private report references and prohibited cache containers. It is a bounded
review aid, not a guarantee that every possible secret can be detected.

## Licensing status

The previous repository did not declare a software license. No new license grant
has been assumed during release preparation; the owner's licensing decision is
pending. Public visibility alone does not establish reuse permission. This also
applies to the newly generated synthetic fixture until a license is chosen.

Third-party dependencies retain their own licenses. No third-party source code
is vendored. The synthetic demo uses no external dataset. The old Yahoo/yfinance
cache has no verified redistribution grant and is not distributed in this tree.
See [the data contract](data-contract.md) for provenance and data-rights details.

Package metadata deliberately includes `Private :: Do Not Upload` to prevent
accidental PyPI submission. It does not change the intended GitHub review flow.
No CI schedule, publication workflow or automatic push is installed.
