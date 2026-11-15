"""Conservative text adapter. Reachy's conversation model resolves dialogue context.

This module strips known conversational wrappers; it is not a general language
model. It never drops object qualifiers or fuzzy-matches medicine names.
"""
import re
import unicodedata


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
    if not medicine and profile.get("medicine_names"):
        return {"status": "NEEDS_CLARIFICATION", "reason": "Use dispense_medicine for this configured medicine box"}
    return {"status": "RESOLVED", "target": target, "description": phrase,
            "category": profile.get("category"), "profile_matched": bool(found),
            "basis": "configured exact name/alias" if found else "description preserved for SAM; no measured profile matched"}
