from src.preprocessing.pipeline import preprocess_text


FIRE_RULE = {
    'rule': 'FIRE_DETECTED',
    'action': 'Fire Fighter required',
    'reason': 'Emergency text contained validated fire-related indicators.',
}
AMBULANCE_RULE = {
    'rule': 'MEDICAL_EMERGENCY',
    'action': 'Ambulance required',
    'reason': 'Emergency text indicated a medical or accident-related incident.',
}
POLICE_RULE = {
    'rule': 'SECURITY_THREAT',
    'action': 'Police required',
    'reason': 'Emergency text described a security or threat scenario.',
}
BLOOD_RULE = {
    'rule': 'BLOOD_REQUIRED',
    'action': 'Blood Donor required',
    'reason': 'Emergency text indicated blood-demand response requirements.',
}
RESCUE_RULE = {
    'rule': 'RESCUE_REQUIRED',
    'action': 'Rescue Team required',
    'reason': 'Emergency text indicated a rescue scenario requiring extraction support.',
}


def evaluate_safety_rules(text: str):
    """Return triggered safety logic and overrides for emergency text."""
    normalized = preprocess_text(text)
    overrides = []
    triggers = []

    if any(token in normalized for token in ['fire', 'smoke', 'burning', 'explosion', 'warehouse fire']):
        overrides.append(FIRE_RULE)
        triggers.append(FIRE_RULE['rule'])
    if any(token in normalized for token in ['medical', 'accident', 'injury', 'heart', 'stroke', 'bleeding', 'ambulance']):
        overrides.append(AMBULANCE_RULE)
        triggers.append(AMBULANCE_RULE['rule'])
    if any(token in normalized for token in ['security', 'threat', 'robbery', 'attack', 'police', 'gun']):
        overrides.append(POLICE_RULE)
        triggers.append(POLICE_RULE['rule'])
    if any(token in normalized for token in ['blood', 'blood donor', 'transfusion']):
        overrides.append(BLOOD_RULE)
        triggers.append(BLOOD_RULE['rule'])
    if any(token in normalized for token in ['rescue', 'trapped', 'collapse', 'flood', 'drowning', 'kidnapped']):
        overrides.append(RESCUE_RULE)
        triggers.append(RESCUE_RULE['rule'])

    return {
        'triggered': triggers,
        'overrides': overrides,
    }
