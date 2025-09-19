from django import forms

class SeedForm(forms.Form):
    seed = forms.CharField(
        required=True,
        label="Seed",
        widget=forms.TextInput(attrs={"placeholder": "Artist - Track"})
    )
    # optional sliders (defaults if missing)
    alpha_emb   = forms.FloatField(required=False, min_value=0.0, max_value=1.0)
    beta_tag    = forms.FloatField(required=False, min_value=0.0, max_value=1.0)
    gamma_pop   = forms.FloatField(required=False, min_value=0.0, max_value=1.0)
    delta_fresh = forms.FloatField(required=False, min_value=0.0, max_value=1.0)

    def clean_seed(self):
        s = (self.cleaned_data.get("seed") or "").strip()
        parts = [p.strip() for p in s.split(" - ", 1)]
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise forms.ValidationError("Please enter in the format: Artist - Track")
        # stash parsed values for easy access
        self.cleaned_data["artist"] = parts[0]
        self.cleaned_data["track"]  = parts[1]
        return s

    @property
    def artist_track(self):
        # Safe to read after is_valid() == True
        return self.cleaned_data.get("artist"), self.cleaned_data.get("track")

    def weights(self):
        # Provide defaults if fields omitted
        return dict(
            ALPHA_EMB = float(self.cleaned_data.get("alpha_emb")   if self.cleaned_data.get("alpha_emb")   is not None else 0.55),
            BETA_TAG  = float(self.cleaned_data.get("beta_tag")    if self.cleaned_data.get("beta_tag")    is not None else 0.25),
            GAMMA_POP = float(self.cleaned_data.get("gamma_pop")   if self.cleaned_data.get("gamma_pop")   is not None else 0.12),
            DELTA_FRESH=float(self.cleaned_data.get("delta_fresh") if self.cleaned_data.get("delta_fresh") is not None else 0.08),
        )