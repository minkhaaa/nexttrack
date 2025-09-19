from rest_framework import serializers

from .models import Track


class TrackSerializer(serializers.ModelSerializer):
    class Meta:
        model = Track
        fields = ["track_id", "title", "artist", "genre", "mood"]


class RecommendationRequestSerializer(serializers.Serializer):
    seed = serializers.CharField()
    alpha_emb = serializers.FloatField(required=False, min_value=0.0, max_value=1.0, default=0.55)
    beta_tag = serializers.FloatField(required=False, min_value=0.0, max_value=1.0, default=0.25)
    gamma_pop = serializers.FloatField(required=False, min_value=0.0, max_value=1.0, default=0.12)
    delta_fresh = serializers.FloatField(required=False, min_value=0.0, max_value=1.0, default=0.08)
    sort = serializers.CharField(required=False, allow_blank=True, default="hybrid")

    _SORT_CHOICES = {"hybrid", "recent", "popular", "duration"}

    def validate(self, attrs):
        seed = (attrs.get("seed") or "").strip()
        parts = [p.strip() for p in seed.split(" - ", 1)]
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise serializers.ValidationError({"seed": "Please enter in the format: Artist - Track"})
        attrs["artist"] = parts[0]
        attrs["track"] = parts[1]
        return attrs

    def validate_sort(self, value):
        norm = (value or "hybrid").strip().lower()
        if norm not in self._SORT_CHOICES:
            choices = ", ".join(sorted(self._SORT_CHOICES))
            raise serializers.ValidationError(f"Sort must be one of: {choices}")
        return norm
