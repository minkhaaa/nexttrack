"""
Django settings:nexttrack_api project.
"""

from pathlib import Path
import os
import environ
import logging

# ---------------------------------------
# Paths
# ---------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

# ---------------------------------------
# Env
# ---------------------------------------
env = environ.Env()
environ.Env.read_env(os.path.join(BASE_DIR, ".env"))  # Explicit path to .env file

# --- Spotify (keep for future use) ---
SPOTIPY_CLIENT_ID     = env("SPOTIPY_CLIENT_ID", default="")
SPOTIPY_CLIENT_SECRET = env("SPOTIPY_CLIENT_SECRET", default="")
SPOTIPY_REDIRECT_URI  = env("SPOTIPY_REDIRECT_URI", default="")

# --- Recommender env knobs (all optional) ---
LASTFM_API_KEY = env("LASTFM_API_KEY", default="48d48554fffd14c776158db8badeae87")
PRE_N          = env.int("PRE_N", default=200)
TOP_K          = env.int("TOP_K", default=30)
ALPHA_EMB      = env.float("ALPHA_EMB", default=0.55)
BETA_TAG       = env.float("BETA_TAG", default=0.25)
GAMMA_POP      = env.float("GAMMA_POP", default=0.12)
DELTA_FRESH    = env.float("DELTA_FRESH", default=0.08)
MMR_LAMBDA     = env.float("MMR_LAMBDA", default=0.6)
MMR_K          = env.int("MMR_K", default=20)

# Optional: control where Sentence-Transformers caches models
SENTENCE_TRANSFORMERS_HOME = env("SENTENCE_TRANSFORMERS_HOME", default=str(BASE_DIR / ".models"))
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", SENTENCE_TRANSFORMERS_HOME)

# ---------------------------------------
# Core
# ---------------------------------------
SECRET_KEY = env("DJANGO_SECRET_KEY", default="dev-insecure-secret")
DEBUG = env.bool("DEBUG", default=True)

ALLOWED_HOSTS = env.list("ALLOWED_HOSTS", default=["*"])

# If deploy behind a domain, set this (esp. if DEBUG=False)
CSRF_TRUSTED_ORIGINS = env.list(
    "CSRF_TRUSTED_ORIGINS",
    default=[
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
)

# ---------------------------------------
# Apps
# ---------------------------------------
INSTALLED_APPS = [
    # Django
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",

    # Third-party
    "rest_framework",

    # Tailwind (provides {% tailwind_css %} via tailwind_tags)
    "tailwind",
    "theme",  # created by `python manage.py tailwind init`
    # Optional during dev: auto-reload tailwind
    "django_browser_reload" if DEBUG else None,

    # Local app
    "recommender",
]
# Remove None entries if DEBUG is False
INSTALLED_APPS = [a for a in INSTALLED_APPS if a]

# Tailwind settings
TAILWIND_APP_NAME = "theme"
INTERNAL_IPS = ["127.0.0.1"]

# ---------------------------------------
# Middleware
# ---------------------------------------
MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]
if DEBUG:
    MIDDLEWARE.append("django_browser_reload.middleware.BrowserReloadMiddleware")

# ---------------------------------------
# URLs / WSGI
# ---------------------------------------
ROOT_URLCONF = "nexttrack_api.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "nexttrack_api.wsgi.application"

# ---------------------------------------
# Database (SQLite for dev)
# ---------------------------------------
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": str(BASE_DIR / "db.sqlite3"),
    }
}

# ---------------------------------------
# Password validators (kept as-is)
# ---------------------------------------
AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

# ---------------------------------------
# I18N
# ---------------------------------------
LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_L10N = True
USE_TZ = True

# ---------------------------------------
# Static / Media
# ---------------------------------------
STATIC_URL = "/static/"

# Where manage.py collectstatic will put files (for prod)
STATIC_ROOT = BASE_DIR / "staticfiles"

# Where project-level static files live (for dev)
STATICFILES_DIRS = [
    BASE_DIR / "static",  
]

# Optional media
MEDIA_URL = "/media/"
MEDIA_ROOT = BASE_DIR / "media"

# ---------------------------------------
# DRF (kept minimal)
# ---------------------------------------
REST_FRAMEWORK = {
    "DEFAULT_RENDERER_CLASSES": [
        "rest_framework.renderers.JSONRenderer",
        "rest_framework.renderers.BrowsableAPIRenderer" if DEBUG else "rest_framework.renderers.JSONRenderer",
    ],
}
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"
# ---------------------------------------
# Logging (dev-friendly)
# ---------------------------------------
LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "handlers": {"console": {"class": "logging.StreamHandler"}},
    "root": {"handlers": ["console"], "level": "INFO"},
}
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)