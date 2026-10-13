# Strata startup fix, 2026-10-13

The local coordinator rejected five successive Strata starts because it required
16 GiB of available Windows RAM. The host had about 14 GiB available after WSL
cache cleanup. Strata's existing budgeted engine can shrink its expert cache to
available physical RAM and commit capacity while reserving 4 GiB of headroom.

The coordinator now permits startup above an 8 GiB floor, retains the GPU and
low-memory checks, logs available RAM, and returns specific startup errors.
The reservation client preserves those errors, closes HTTP error responses,
and still cancels failed acquisitions. Unexpected coordinator errors retain the
generic response.

The installed expert-cache cap was reduced from 8 to 4 GiB after a working
8 GiB-cache run left only about 2 GiB of free system RAM during inference.
A stale coordinator process was also removed and the scheduled coordinator
started with the updated code.

## Runtime files outside this repository

[strata-startup-fix.patch](strata-startup-fix.patch) records the already-applied
changes to these paths relative to `C:/Server/stacks`:

- `local-ai/gpu_coordinator.py`
- `local-ai/test_gpu_coordinator.py`
- `strata/strata-coder-iq1_m.json`

The helper directory is not a Git repository, and the installed model config is
ignored by Strata's repository. The patch preserves their changes without adding
credentials, generated model files, or another repository. It is an archival
patch, not an automatic deployment; it must not be applied again to this host.
Its reverse application was checked against the installed files with
`git apply --reverse --check --ignore-whitespace` to tolerate Windows CRLF context.

## Verification completed before the harness A/B test

- 17 coordinator/helper tests, 12 reservation-client tests, and 6 application
  reservation integration tests passed.
- Two fresh model starts with the final 4 GiB cache each returned
  `STRATA_STARTUP_OK` and released the GPU. Loading took 28.5 and 19.4 seconds.
- Sampling Windows available RAM every two seconds during both checks recorded
  a minimum of 7.09 GiB; the 4 GiB margin assertion passed.
- Strata also completed the previously blocked, condensed harness review using
  its earlier 8 GiB cache: 588 input tokens and 293 output tokens in 130 seconds
  of inference. Its suggestions remain subject to code review and live A/B tests.

Long coding jobs have not been benchmarked with the smaller cache; replies may
be slower. The host helper uses the updated reservation client directly from
source. The audiobook container has not been rebuilt with that diagnostic
client change. No harness implementation changes are included in this fix.
