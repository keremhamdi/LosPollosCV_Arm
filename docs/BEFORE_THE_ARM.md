# Before the arm is available

Everything here runs on a laptop. Nothing below moves a motor.

## 1. Check the software

```bash
python -m pytest -q
```

## 2. Check the conversation with Medicine Buddy

With the Reachy team's repository cloned next to this one:

```bash
python examples/simulate_reachy.py /path/to/LosPollosForTheElders
```

It runs their real Buddy server against this arm service and prints what the
arm received, what it understood and what Buddy would say.

## 3. Fill in the items file

Add each medicine and everyday object to `items_v1.json`, then:

```bash
python reachy_target.py items_v1.json
```

It lists what is still missing per item (measurements, medicine `form`,
`fetch_allowed`). `fetch_allowed` stays off until the item has been gripped on
the real arm, so that line is expected for now.

An everyday object may list other phrases for SAM 3 to try:

```json
"water bottle": {"height_cm": 22.0, "width_cm": 6.5, "grasp": "side",
                 "aliases": ["drinking bottle"],
                 "sam_prompts": ["plastic bottle", "clear bottle"]}
```

`aliases` are other things the person may say. `sam_prompts` are other things
SAM 3 may look for; the best score among them counts. Medicines ignore
`sam_prompts`: they are found by one description and confirmed by their label.

## 4. Check SAM 3 on photos

Take photos of the table from where the standing camera will be, then:

```bash
python target_selector_v5.py --image table.jpg --target "water bottle" --category bottle --vlm none
```

Try every item, every form of medicine, and scenes with two similar objects.

## 5. Run the service without hardware

```bash
python arm_service.py --fake --host 0.0.0.0 --port 9090
```

```bash
curl -s http://localhost:9090/health
```

The Reachy team can point `MEDICINE_BUDDY_ARM_URL` at this machine and speak a
request; `curl -s http://localhost:9090/debug/last` then shows what arrived.

## Needs the arm

Calibration, workspace and destination measurements, gripping each item,
`fetch_allowed`, blister pickup, label reading under the real lighting, and
`--enable-motion`.
