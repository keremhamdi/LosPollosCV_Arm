"""Conservative text adapter. Reachy's conversation model resolves dialogue context.

This module strips known conversational wrappers; it is not a general language
model. It never drops object qualifiers or fuzzy-matches medicine names.
"""
import re
import unicodedata


# Physical forms a medicine may take on the table -> broad SAM prompt for that form.
# "pill container" is a labelled bottle/tub of pills. Loose pills carry no readable
# name, so their identity cannot be checked and they are never picked.
MEDICINE_FORMS = {"box": "box", "blister": "blister pack", "pill container": "pill bottle"}


def normalize(text):
    text = unicodedata.normalize("NFKC", text).replace("’", "'").casefold()
    return " ".join(re.sub(r"[^\w']+", " ", text).replace("_", " ").split())


def resolve_target(text, items, *, medicine=False):
    if not isinstance(text, str) or not text.strip() or len(text) > 500:
        return {"status": "NEEDS_CLARIFICATION", "reason": "Provide one object description, at most 500 characters"}
    phrase = normalize(text)
    if re.search(r"\b(knife|knives|scissors|blade|razor|syringe|needle|hot|boiling|weapon|gun|person|people)\b", phrase):
        return {"status": "REFUSED", "reason": "Hot items, sharps, weapons and moving people are outside this fetch workflow"}
    if re.search(r"\b(no|not|don't|dont|never|stop|cancel|ignore|instead)\b", phrase):
        return {"status": "NEEDS_CLARIFICATION", "reason": "Negation, correction or cancellation needs a new confirmed request"}
    if re.search(r"\b(and|or|then|both|either|two|three|four)\b", phrase):
        return {"status": "NEEDS_CLARIFICATION", "reason": "Request exactly one object"}
    if not medicine:
        # Only remove wrappers with known meaning, never arbitrary leading text.
        prefixes = ("hey reachy ", "reachy ", "yes ", "please ", "could you ", "can you ",
                    "would you ", "will you ", "mind ", "kindly ", "i would like ", "i'd like ",
                    "i want ", "bringing me ", "bring me ", "give me ", "get me ", "fetch me ",
                    "find me ", "hand me ", "pick up ", "bring ", "fetch ", "find ", "get ",
                    "the ", "a ", "an ")
        while True:
            prefix = next((p for p in prefixes if phrase.startswith(p)), None)
            if prefix is None:
                break
            phrase = phrase[len(prefix):]
        suffixes = (" please", " for me", " thank you", " thanks", " from the table", " on the table")
        while True:
            suffix = next((s for s in suffixes if phrase.endswith(s)), None)
            if suffix is None:
                break
            phrase = phrase[:-len(suffix)]
        if (not phrase or len(phrase.split()) > 12 or
            re.search(r"\b(it|this|that|one|something|anything|yes|yeah|ok|okay|thanks|because|want|need|could|bring|give|fetch|please)\b", phrase) or
            re.search(r"\b(next to|beside|behind|closest|nearest|left|right)\b", phrase)):
            return {"status": "NEEDS_CLARIFICATION", "reason": "Reachy must resolve the reference to a short, unambiguous object description"}
    found = []
    for target, profile in items.items():
        aliases = profile.get("medicine_names", []) if medicine else [target, *profile.get("aliases", [])]
        if any(phrase == normalize(alias) for alias in aliases):
            found.append(target)
    if len(found) > 1:
        return {"status": "NEEDS_CLARIFICATION", "reason": "Alias matches multiple configured objects", "choices": found}
    if medicine and not found:
        return {"status": "NEEDS_CLARIFICATION", "reason": "Medicine has no explicit physical-box mapping in medicine_names"}
    target = found[0] if found else phrase
    profile = items.get(target, {})
    form = profile.get("form")
    if medicine:
        if form == "loose pills":
            return {"status": "REFUSED", "reason": "Loose pills have no readable name; keep them in a labelled box, blister or pill container"}
        if form not in MEDICINE_FORMS:
            return {"status": "NEEDS_CLARIFICATION", "reason": f"Medicine item needs form: one of {', '.join(MEDICINE_FORMS)}"}
        if form == "blister" and profile.get("grasp") != "top":
            return {"status": "NEEDS_CLARIFICATION", "reason": "A flat blister must be configured with grasp: top"}
    if not medicine and profile.get("medicine_names"):
        return {"status": "NEEDS_CLARIFICATION", "reason": "Use dispense_medicine for this configured medicine box"}
    # Extra SAM phrases widen recognition of everyday objects only. A medicine is
    # found by its one configured description and confirmed by reading its label.
    extra = [] if medicine else profile.get("sam_prompts", [])
    if (not isinstance(extra, list) or len(extra) > 5 or
            any(not isinstance(p, str) or not 0 < len(normalize(p).split()) <= 6 for p in extra)):
        return {"status": "NEEDS_CLARIFICATION", "reason": "sam_prompts must be at most five short phrases"}
    return {"status": "RESOLVED", "sam_prompts": extra, "target": target, "description": phrase,
            "category": profile.get("category") or (MEDICINE_FORMS[form] if medicine else None),
            "form": form if medicine else None, "profile_matched": bool(found),
            "basis": "configured exact name/alias" if found else "description preserved for SAM; no measured profile matched"}


def check_items(items):
    """Problems in an items file, and what Reachy can ask for. No hardware needed."""
    problems, names = [], {}
    for target, profile in items.items():
        medicine = bool(profile.get("medicine_names"))
        for spoken in profile.get("medicine_names", []) if medicine else [target, *profile.get("aliases", [])]:
            out = resolve_target(spoken, items, medicine=medicine)
            if out["status"] != "RESOLVED" or out["target"] != target:
                problems.append(f"{target}: '{spoken}' -> {out['status']}: {out.get('reason', 'resolves to ' + str(out.get('target')))}")
            names.setdefault(normalize(spoken), []).append(target)
        if not medicine and profile.get("form"):
            problems.append(f"{target}: has a medicine form but no medicine_names")
        if not all(isinstance(profile.get(k), (int, float)) and profile[k] > 0 for k in ("height_cm", "width_cm")):
            problems.append(f"{target}: needs measured height_cm and width_cm")
        if profile.get("fetch_allowed") is not True:
            problems.append(f"{target}: fetch_allowed is not true, so the arm will preview but never pick it")
    return problems


if __name__ == "__main__":
    import json, sys
    found = check_items(json.load(open(sys.argv[1] if len(sys.argv) > 1 else "items_v1.json")))
    print("\n".join(found) or "Items file is ready")
    sys.exit(1 if found else 0)
