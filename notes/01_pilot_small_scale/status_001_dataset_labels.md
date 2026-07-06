# Status 001: GeoPrivacy Q7 Label Preparation

Date: 2026-07-01

## Completed

Created the first processed benchmark label table:

- File: data/processed/geoprivacy_q7_labels.csv
- Total samples: 1200
- Train samples: 718
- Validation samples: 239
- Test samples: 243

## Label distribution

- A: 486
- B: 241
- C: 473

## Privacy-sensitive setup

For the first-stage over-disclosure project:

- Privacy-sensitive inputs: A and B
- Disclosure-allowed inputs: C

Counts:

- privacy_sensitive = 1: 727
- privacy_sensitive = 0: 473

## Current limitation

Images are not downloaded yet.

Current image status:

- Images found: 0 / 1200

This is expected because the benchmark repository does not include images directly.
The next step is to download images into:

external/VLM-GeoPrivacyBench/data/images/
