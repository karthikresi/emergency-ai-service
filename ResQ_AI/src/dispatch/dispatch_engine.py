class DispatchEngine:
    def __init__(self, config=None):
        base = {
            'initial_radius_km': 1,
            'second_radius_km': 5,
            'third_radius_km': 10,
            'max_radius_km': 10,
            'max_notifications': 2,
            'escalation_threshold': 1,
            'skill_match_weight': 40,
            'availability_weight': 20,
            'distance_weight': 15,
            'eta_weight': 12,
            'reliability_weight': 18,
            'experience_weight': 10,
            'acceptance_weight': 10,
        }
        if config:
            base.update(config)
        self.config = base

    def expand_radius(self, current_radius):
        radius_steps = [self.config['initial_radius_km'], self.config['second_radius_km'], self.config['third_radius_km'], self.config['max_radius_km']]
        for radius in radius_steps:
            if radius > current_radius:
                return radius
        return self.config['max_radius_km']

    def estimate_eta(self, distance_km):
        return max(1.0, distance_km * 4.0)

    def rank_responders(self, responders, required_skills):
        required = {str(skill).lower() for skill in (required_skills or [])}
        ranked = []
        for responder in responders:
            if not responder.get('available', True):
                continue
            skill = str(responder.get('skill', '')).lower()
            score = 0.0
            if required and skill in required:
                score += self.config.get('skill_match_weight', 40)
            score += self.config.get('availability_weight', 20) * (1 if responder.get('available', True) else 0)
            distance = float(responder.get('distance_km', 0.0) or 0.0)
            score += max(0.0, self.config.get('distance_weight', 15) * (10 - min(distance, 10)))
            eta = float(responder.get('eta_min', self.estimate_eta(distance)) or self.estimate_eta(distance))
            score += max(0.0, self.config.get('eta_weight', 12) * (20 - min(eta, 20)))
            score += self.config.get('reliability_weight', 18) * float(responder.get('reliability', 0.0) or 0.0)
            score += self.config.get('experience_weight', 10) * min(float(responder.get('experience_years', 0.0) or 0.0) / 10.0, 1.0)
            score += self.config.get('acceptance_weight', 10) * float(responder.get('acceptance_probability', 0.0) or 0.0)
            ranked.append({
                'id': responder.get('id', 'unknown'),
                'skill': responder.get('skill', 'Unknown'),
                'available': responder.get('available', True),
                'distance_km': distance,
                'eta_min': eta,
                'score': round(score, 3),
            })
        ranked.sort(key=lambda item: item['score'], reverse=True)
        return ranked

    def dispatch(self, responders, required_skills, current_radius_km=1.0):
        candidates = self.rank_responders(responders, required_skills)
        selected = candidates[: min(self.config.get('max_notifications', 2), len(candidates))]
        decision = {
            'current_radius_km': current_radius_km,
            'status': 'ready' if selected else 'escalate',
            'responders': selected,
        }
        return decision
