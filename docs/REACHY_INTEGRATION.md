# Reachy Mini → text → SAM → arm result

The Reachy conversation app already has a language model. It resolves a spoken
sentence to one concise description, asks for confirmation, then sends JSON.
The arm PC runs `arm_service.py` on **9090**, backed by the existing two-camera
controller on **8890**. No Ollama or additional language-model server is needed.

Example: “Could you please bring me the small blue water bottle?” becomes
`object: "small blue water bottle"`. Colour and size stay in the SAM prompt.
Reachy must clarify “it”, alternatives, corrections or unresolved references
before confirming. The arm adapter also strips common polite wrappers, supports
explicit configured aliases and refuses uncertain wording. It is not a general
language model and does not infer arbitrary conversational context itself.

The included [Reachy patch](../reachy_patch/README.md) targets
`tommasosommaruga/LosPollosForTheElders` main `3ab8321`. Apply and test it there,
then deploy that updated Reachy app using its normal install procedure. This
repository does not deploy or operate the voice robot automatically.

## 1. Check the protocol without cameras or motors

From this repository root:

```bash
.venv/bin/python -B arm_service.py --fake --host 0.0.0.0
```

The default bind address without `--host` is localhost. Use the LAN binding only
on the trusted robot network: the v1 protocol has no client authentication and
the arm trusts Buddy's voice guard for spoken consent. Do not expose it publicly.

On the Reachy host, set:

```bash
MEDICINE_BUDDY_ARM_URL=http://ARM_PC_IP:9090
MEDICINE_BUDDY_DRY_RUN=1
MEDICINE_BUDDY_ARM_TIMEOUT_S=180
MEDICINE_BUDDY_VOICE_GUARD=1
MEDICINE_BUDDY_REQUIRE_PASSPHRASES=1
```

`GET /health` reports configuration. `--fake` only exercises text/protocol
handling; it never claims to have detected or placed an object. An empty Buddy
arm URL retains the existing medicine FakeArm path; the new object fetch tool
returns an explicit simulated preview after confirmation.

Buddy's existing Family guardrails setting also controls its voice guard and
overrides the environment when saved. Verify guardrails are enabled in Family
settings and the Debug page reports `voice_guard.enabled: true`. The patch blocks
live fetch/dispense when that guard is disabled; a tool's `confirmed:true` alone
is not evidence that the person said yes.

## 2. Run real SAM using both existing cameras

Start the updated controller in read-only mode using [Linux setup](../LINUX_HANDOFF.md).
Then start the service in another terminal:

```bash
.venv/bin/python -B arm_service.py --host 0.0.0.0 \
  --arm-url http://localhost:8890 --items items_v1.json \
  --workspace config/local/workspace.json
```

Keep Buddy in dry-run. The service has its own motion opt-in, so even a mistaken
`dry_run: false` cannot move the arm with the command above. Each request runs
SAM on one standing/hand pair and saves masks, input images and reports under
`review_outputs/reachy/`. Both cameras remain continuously connected.

For an offline two-photo test use `--image /path/stand.jpg --hand-image /path/hand.jpg`
and the matching image-search workspace. This cannot be combined with motion.

A manual detection-only smoke test from the arm PC (Reachy will send these
requests itself once its patch is deployed):

```bash
.venv/bin/python - <<'PY'
import requests, uuid
body = {"schema": "medicine_buddy.arm.v1", "message_type": "locate",
        "request_id": str(uuid.uuid4()), "one_shot_detect": True, "dry_run": True,
        "object": "Could you please bring me the white cup from the table?"}
print(requests.post("http://localhost:9090/v1/locate", json=body, timeout=180).json())
PY
```

## Protocol

All POSTs use `Content-Type: application/json`, `schema: "medicine_buddy.arm.v1"`,
a fresh UUID `request_id`, `one_shot_detect: true` and a real JSON boolean `dry_run`.
The response comes back on the same HTTP request. There are no callbacks.

| Endpoint | Target fields | Purpose |
| --- | --- | --- |
| `/v1/fetch` | `message_type: "fetch"`, `object: "small blue water bottle"`, `deliver_to: "handover"` | Household object request after yes |
| `/v1/move_object` | `message_type: "move_object"`, `object: {"label": "water bottle", "kind_hint": "bottle"}`, `destination: "bowl"` or `"handover"`, `actions: ["locate","pick","place"]` | What Medicine Buddy main (ba644ff, `move_object` tool) sends today; needs no Reachy patch. On failure `error` is a speakable sentence and `error_code` the machine code |
| `/v1/dispense` | `message_type: "dispense"`, `medicine: {"id":12,"name":"Aspirin","person":"Mario"}`, `slot_hint:null`, `ocr_min_confidence:0.65`, `actions:["locate","pick_to_bowl","verify_place"]` | Configured closed medicine box after yes |
| `/v1/locate` | `message_type: "locate"`, `object: "white cup"` | Detection-only extension; never moves, including with dry_run=false |
| `/v1/check_taken` | `message_type: "check_taken"`, `medicine: {...}` | Returns `taken:null`; Buddy must ask the person |

`object` is the resolved description. Do not pass raw conversation history as the
medicine name. For medicines, an exact entry in an item's **`medicine_names`** list
maps the database name to its physical SAM description. Ordinary `aliases` apply
to household objects. Duplicate aliases/mappings and unknown medicines are
rejected. There is no fuzzy medicine matching and no automatic mapping from an
arbitrary pill box to Aspirin. Configure that association only from the real box.
The optional `slot_hint` is not used as a measured slot assignment; `slot_id`
remains null in this vision-based route.

An unresolved description returns `error: "NEEDS_CLARIFICATION"`. A successful
detection is separate from successful delivery. Responses include `resolution`
(the exact SAM target), `vision.selection`, and reasons for blocked motion.
Household selection scores remain under `vision.selection`; `score` on a medicine
delivery is the minimum pre-pick standing-camera OCR word confidence, not SAM confidence.

For dispense/fetch, `ok:true`, `final_state:"placed"`, `place_verified:true`
require a live pick, controller release and passing post-release camera check.
Dry-runs return `ok:false`, `final_state:"aborted"`, `error:"DRY_RUN"`,
`preview_only:true`. The wire contract has no `preview` final state, so the patched
fetch tool speaks a preview explicitly. A medicine dry-run with a remote service
does not log a real Placed/Taken event; the existing orchestrator treats the
unsuccessful delivery as Failed. Fake-mode responses also never claim placement.

## 3. Configure physical delivery before enabling motion

First complete [workspace and calibration validation](WORKSPACE_AND_VALIDATION.md).
The known saved calibration still fails its error gate. An unchanged setup does
not fix those errors. Every item needs verified dimensions and a supported grasp
method; add `"fetch_allowed": true` only after assessing that object and route.
Use rigid, fitting objects. Medicines use the dispense route and labelled closed
boxes; the service does not handle dosing, loose pills or packaging operations.

Create `config/local/destinations.json` with `handover` and/or `bowl`. Each entry
contains the same `schema_version:1`, `role:"stand"`, `camera_id`, `image_size` and
convex `polygon_px` fields as a workspace, plus:

- `target_px`: the calibrated table-plane pixel for the taught destination point.
- `release_height_cm` and `above_cm`: measured release/approach heights for this
  object and destination. Approach must be at least as high as the pick approach.
- `measurements_verified:true`: only after checking the real destination,
  container rim, object/finger clearance and approach route.

These are geometry measurements, not values inferred from the photographs.
The destination must be inside the independently validated camera/table regions.
Its image polygon must cover the expected visible object after placement and
must be empty of matching target proposals before picking. The controller checks
destination IK before picking; that check is not obstacle avoidance.

For medicine label checks, install the system Tesseract English OCR engine on
Linux (`sudo apt install tesseract-ocr tesseract-ocr-eng` on Debian/Ubuntu).
Exact label tokens must pass at least 0.65 confidence in the **standing** selected mask
before pickup, and again in the standing view after placement. Wrist OCR is
supplementary when a unique target is visible; it cannot replace the standing
label check, and the wrist need not see the target before approach. A higher request
threshold is honoured; a lower one cannot weaken the 0.65 floor. Unavailable or
uncertain OCR blocks delivery. This is a package-label check, not verification
of contents, formulation or a prescription. See [Tesseract's official TSV documentation](https://tesseract-ocr.github.io/tessdoc/Command-Line-Usage.html#tsv-output).

Only for a calibrated physical trial, run the controller without read-only and
start the service using the verified local files:

```bash
.venv/bin/python -B arm_service.py --host 0.0.0.0 \
  --arm-url http://localhost:8890 --items config/local/items.json \
  --workspace config/local/workspace.json --validation config/local/validation.json \
  --destinations config/local/destinations.json --enable-motion
```

Buddy must also have dry-run disabled
and receive a fresh spoken yes. No hardware was operated while implementing this.

After release the service requires a new camera pair, unchanged camera/geometry,
one eligible target at the destination, at least 90% of its mask within that
region, and little overlap with its original location. This is conservative
**2D placement evidence**, not proof that it sits inside a bowl in 3D. Occlusion
can produce an unverified result. No camera test here proves ingestion.

## Timeouts, duplicates and debugging

Request IDs and results persist in `config/local/reachy_requests.sqlite3`. Repeating
the same ID/content returns its recorded result; different content under that ID
is rejected. Active/interrupted jobs, motion timeouts and unverified placement
after release block new jobs, including requests with fresh IDs. No motion request is
automatically retried. After a lost response, check the physical arm and journal.
Only after operator review, retire an unresolved request locally with:

```bash
.venv/bin/python -B arm_service.py --acknowledge-unknown THE_REQUEST_UUID
```

Do this while the service is stopped; it never replays that request or moves the
arm. Keep the journal across restarts. Do not delete it to retry a pickup.
The service's `GET /debug/last` is localhost-only. The patched Reachy Family Debug
page also shows the last structured arm request/response, with its normal access
control. Local journals, calibration, patient data and photos are excluded from
Git and the source archive.

## Medicine forms

Every item that Buddy may dispense needs `medicine_names` (the exact names in
Buddy's medicine list) and a `form` in the items file:

| `form` | What it is | Notes |
|---|---|---|
| `box` | Rectangular carton with the name printed on it | Name is read by OCR on the standing camera |
| `blister` | Thin aluminium strip | Must have `"grasp": "top"` and a measured `height_cm`; the name must be readable on the side facing up |
| `pill container` | Labelled bottle or tub of pills | Label must face the standing camera |
| `loose pills` | Pills lying on the table | Always refused: there is no name to read, so identity cannot be checked |

```json
"paracetamol box": {"medicine_names": ["Paracetamol"], "form": "box",
                    "height_cm": 0, "width_cm": 0, "grasp": "top", "fetch_allowed": false}
```

Replace the zeros with measured values and set `fetch_allowed` to `true` only
after a physical trial of that item. A medicine with no `form`, or a blister not
set to a top grasp, is answered with `NEEDS_CLARIFICATION` and nothing moves.
The form also supplies the broad SAM prompt (`box`, `blister pack`,
`pill bottle`) when the item has no `category`. OCR on reflective foil and
picking a strip a few millimetres high have not been tried on the real arm.
