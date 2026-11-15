# Note for the Reachy team

Checked against LosPollosForTheElders main `4070f17` (v0.9.4).

**No code change is needed on your side.** Our arm service accepts what Buddy
already sends on `/v1/dispense`, `/v1/move_object` and `/v1/check_taken`.

To connect:

1. Put the Reachy on the same network as the arm PC.
2. Set `MEDICINE_BUDDY_ARM_URL=http://<arm-pc-ip>:9090`.
3. Do not run `integrations/arm/arm_listener.py`; our service replaces it.

What to expect:

- `ok: true` only comes back after a camera-verified placement.
- When the arm did not act, `error` is a sentence you can speak and
  `error_code` is the machine code (`DRY_RUN`, `NEEDS_CLARIFICATION`,
  `UNSUPPORTED_OBJECT`, `UNSUPPORTED_DESTINATION`, ...).
- `destination` may be `bowl` or `handover`.
- `check_taken` always answers `taken: null`; ask the person.
- Medicine names must match our `medicine_names` exactly. Send us your list.
- Buddy now always sends `dry_run: false`. The arm still only moves when we
  start our service with motion enabled.

Optional on your side:

- A preview reply (`error_code: DRY_RUN` or `FAKE_MODE`) is currently recorded
  as a failed dose and spoken as "could not bring". Treating it as a preview
  would keep test runs out of the dose history.
- `move_object` reports success on `ok` and `final_state` alone. Also checking
  `place_verified` and that `request_id` matches would guard against other
  arm services.
