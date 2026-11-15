# Patch for the Reachy team

`fetch_object.patch` applies to
[LosPollosForTheElders](https://github.com/tommasosommaruga/LosPollosForTheElders)
main **3ab8321**, inspected on 2026-11-15. It adapts the earlier supplied fetch
patch to the current arm service. It is included here for review/application;
the friend's repository and robot installation have not been changed remotely.

In a clean checkout of that repository:

```bash
git switch -c feature/reachy-sam-fetch
git apply --check /path/to/LosPollosCV_Arm/reachy_patch/fetch_object.patch
git apply /path/to/LosPollosCV_Arm/reachy_patch/fetch_object.patch
cd apps/medicine_buddy
python -m pip install -e '.[dev]' fastapi httpx httpx2 uvicorn websockets
python -m pytest -q
```

If upstream has moved, review the failing hunks; do not force the patch onto a
different implementation. The package version becomes **0.9.1**. The patch adds:

- `FetchCommand`, `FetchResult`, `HttpArmClient.fetch()` and `/arm/fetch`.
- The `fetch_object` conversation tool and profile registration, with instructions
  to extract one concise object description while preserving identifying details.
- The existing ask-first/real-spoken-yes guard for fetch. Live fetch and dispense
  are blocked if Family settings have disabled the guard. No flag from the model
  alone is accepted as spoken consent when the guard is enabled.
- Environment-controlled dry-run in `main.py` (still enabled by default).
- Explicit preview wording, verified-placement checks and request-ID/schema
  validation for arm responses. Unknown or unsuccessful delivery cannot become
  an affirmative success report through this HTTP client.
- Last structured arm request/response on the existing protected Family Debug
  page. It does not add raw voice transcripts to those records.
- Fake preview for household fetch with an empty arm URL; existing medicine
  FakeArm demo behaviour is preserved.

Validation: the unchanged source passed **394 tests**; the patched source passed
**403 tests** using Python 3.12 and the installed Reachy Mini **1.14.0** SDK, with
FastAPI/WebSocket test dependencies in an isolated environment. The actual
`HttpArmClient` was also exercised against this repository's HTTP service with a
fake vision/motor backend. No speech session or robot movement was performed.
The pinned conversation application was not launched or installed in that test
environment; install the full app using the project's normal procedure before
deployment. Do not interpret protocol tests as proof the live conversation model
will always extract the right referent or call the right tool.

After the Reachy team has reviewed/applied/pushed its patch, deploy using that
repository's `scripts/install_on_robot.sh [host] [ref]`. Uncommitted local edits
are not installed on Reachy. Set the arm URL to port **9090**, begin in dry-run,
and follow [the integration guide](../docs/REACHY_INTEGRATION.md). Preserve the
robot's existing account keys and settings.
