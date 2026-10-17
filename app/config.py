"""Static configuration: filesystem paths, voices, model table, phase
metadata and the script-writer tables (local + cloud).

Pure data, no behaviour. Imported by every other module.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

# ----------------------------------------------------------------- paths ---
BASE_DIR = Path(__file__).resolve().parent.parent   # project root (server.py lives here)
STATIC_DIR = BASE_DIR / "static"
OUTPUTS_DIR = BASE_DIR / "outputs"
UPLOADS_DIR = BASE_DIR / "uploads"
OUTPUTS_DIR.mkdir(exist_ok=True)
UPLOADS_DIR.mkdir(exist_ok=True)

ERROR_LOG = BASE_DIR / "errors.log"
CALIB_PATH = BASE_DIR / "calib.json"
KEYS_PATH = BASE_DIR / "keys.json"

# ---------------------------------------------------------------- voices ---
# From official README (Qwen/Qwen3-TTS-12Hz-*-CustomVoice)
VOICES: List[Dict[str, Any]] = [
    {
        "id": "Aiden", "label": "Aiden", "gender": "Male", "age": "Young Adult",
        "native_language": "English", "region": "American",
        "description": "Sunny American male voice with a clear midrange.",
        "tags": ["american", "male", "english", "narration", "podcast", "high-quality"],
        "featured": True, "flag": "🇺🇸",
        "best_for": "Podcasts, narration, ads, YouTube — flagship US voice",
    },
    {
        "id": "Ryan", "label": "Ryan", "gender": "Male", "age": "Young Adult",
        "native_language": "English", "region": "American",
        "description": "Dynamic male voice with strong rhythmic drive.",
        "tags": ["american", "male", "english", "energetic", "promo", "high-quality"],
        "featured": True, "flag": "🇺🇸",
        "best_for": "Trailers, promos, energetic reads, storytelling",
    },
    {
        "id": "Vivian", "label": "Vivian", "gender": "Female", "age": "Young",
        "native_language": "Chinese", "region": "Chinese",
        "description": "Bright, slightly edgy young female voice.",
        "tags": ["female", "chinese", "bright", "young"],
        "featured": False, "flag": "🇨🇳",
        "best_for": "Multilingual content, character work",
    },
    {
        "id": "Serena", "label": "Serena", "gender": "Female", "age": "Young",
        "native_language": "Chinese", "region": "Chinese",
        "description": "Warm, gentle young female voice.",
        "tags": ["female", "chinese", "warm", "gentle"],
        "featured": False, "flag": "🇨🇳",
        "best_for": "Audiobooks, calm narration",
    },
    {
        "id": "Uncle_Fu", "label": "Uncle Fu", "gender": "Male", "age": "Middle-aged",
        "native_language": "Chinese", "region": "Chinese",
        "description": "Seasoned male voice with a low, mellow timbre.",
        "tags": ["male", "chinese", "deep", "mellow", "documentary"],
        "featured": False, "flag": "🇨🇳",
        "best_for": "Documentary, deep narration",
    },
    {
        "id": "Dylan", "label": "Dylan", "gender": "Male", "age": "Young",
        "native_language": "Chinese (Beijing Dialect)", "region": "Chinese",
        "description": "Youthful Beijing male voice with a clear, natural timbre.",
        "tags": ["male", "chinese", "beijing", "dialect"],
        "featured": False, "flag": "🇨🇳",
        "best_for": "Beijing dialect content",
    },
    {
        "id": "Eric", "label": "Eric", "gender": "Male", "age": "Young",
        "native_language": "Chinese (Sichuan Dialect)", "region": "Chinese",
        "description": "Lively Chengdu male voice with a slightly husky brightness.",
        "tags": ["male", "chinese", "sichuan", "dialect", "lively"],
        "featured": False, "flag": "🇨🇳",
        "best_for": "Sichuan dialect content",
    },
    {
        "id": "Ono_Anna", "label": "Ono Anna", "gender": "Female", "age": "Young",
        "native_language": "Japanese", "region": "Japanese",
        "description": "Playful Japanese female voice with a light, nimble timbre.",
        "tags": ["female", "japanese", "playful", "anime"],
        "featured": False, "flag": "🇯🇵",
        "best_for": "Anime, playful characters",
    },
    {
        "id": "Sohee", "label": "Sohee", "gender": "Female", "age": "Young",
        "native_language": "Korean", "region": "Korean",
        "description": "Warm Korean female voice with rich emotion.",
        "tags": ["female", "korean", "warm", "emotional"],
        "featured": False, "flag": "🇰🇷",
        "best_for": "K-drama style, emotional reads",
    },
]

LANGUAGES = ["Auto", "English", "Chinese", "Japanese", "Korean", "French",
             "German", "Spanish", "Portuguese", "Russian", "Italian"]

MODELS = {
    "custom-0.6B": {"hf_id": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice", "label": "CustomVoice 0.6B — Fast", "size": "0.6B", "kind": "CustomVoice", "instruct": False, "quality": "Fast / lighter"},
    "custom-1.7B": {"hf_id": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", "label": "CustomVoice 1.7B — High Quality ⭐", "size": "1.7B", "kind": "CustomVoice", "instruct": True, "quality": "Best quality"},
    "design-1.7B": {"hf_id": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign", "label": "VoiceDesign 1.7B — describe any voice", "size": "1.7B", "kind": "VoiceDesign", "instruct": True, "quality": "Creative"},
    "base-0.6B": {"hf_id": "Qwen/Qwen3-TTS-12Hz-0.6B-Base", "label": "Base 0.6B — Voice Clone (fast)", "size": "0.6B", "kind": "Base", "instruct": False, "quality": "Fast clone"},
    "base-1.7B": {"hf_id": "Qwen/Qwen3-TTS-12Hz-1.7B-Base", "label": "Base 1.7B — Voice Clone (best)", "size": "1.7B", "kind": "Base", "instruct": False, "quality": "Best clone"},
}

# ------------------------------------------------------------- phases ---
# Phase metadata: PHASE_CAP is where the live % is capped per phase,
# PHASE_PROGRESS is the base progress per phase; JOB_STEPS/SCRIPT_STEPS are
# the step labels the UI renders.
PHASE_CAP = {"queued": 5, "loading_model": 22, "synthesizing": 99, "saving": 99,
             "done": 100, "error": 99, "cancelled": 100}

PHASE_PROGRESS = {"queued": 0.04, "loading_model": 0.12, "synthesizing": 0.40,
                  "saving": 0.92, "done": 1.0, "error": 1.0}

JOB_STEPS = [["queued", "Queued"], ["loading_model", "Loading model"],
             ["synthesizing", "Synthesizing"], ["saving", "Saving"]]

SCRIPT_STEPS = [["queued", "Queued"], ["loading_model", "Waking writer"],
                ["synthesizing", "Drafting"], ["saving", "Polishing"]]

# ------------------------------------------------------- script writer ---
# FREE local-LLM scripts via Ollama (no cloud, no key).
SCRIPT_MODELS = {
    "best": {"tag": "huihui_ai/qwen3.5-abliterated:9b", "label": "Best — 9B abliterated ⭐",
             "note": "Sharpest prose, unfiltered. Parks TTS voices to fit RAM."},
    "balanced": {"tag": "hermes3:8b", "label": "Balanced — 8B",
                 "note": "Strong quality, lighter footprint."},
    "darkidol": {"tag": "crown/darkidol:latest", "label": "Dark Idol — uncensored",
                 "note": "Roleplay-strong creative writer, unfiltered."},
    "fast": {"tag": "hermes3:3b", "label": "Fast — 3B",
             "note": "Drafts in seconds, simpler prose."},
}
SCRIPT_LENGTHS = {
    "hook": {"label": "Hook ~60 words", "words": 60, "tokens": 160, "eta": 40.0},
    "standard": {"label": "Standard ~180 words", "words": 180, "tokens": 420, "eta": 90.0},
    "extended": {"label": "Extended ~400 words", "words": 400, "tokens": 900, "eta": 180.0},
}
SCENARIOS = {
    "ad": {"label": "📢 Ad Read", "desc": "Punchy 30-second style spot",
           "brief": "Punchy direct-response advertising: magnetic hook, two vivid benefits, hard call to action."},
    "podcast": {"label": "🎧 Podcast Intro", "desc": "Warm host welcome + tease",
                "brief": "Warm podcast host welcoming listeners back, teasing today's topic, setting an inviting tone."},
    "story": {"label": "📖 Short Story", "desc": "Micro-fiction with a twist",
              "brief": "Vivid micro-fiction with a twist ending and rich sensory detail."},
    "dialogue": {"label": "💬 Dialogue", "desc": "Two voices, labeled A:/B:",
                 "brief": "A two-character dialogue with lines labeled A: and B:, distinct voices, natural interruptions and subtext."},
    "monologue": {"label": "🎭 Monologue", "desc": "One raw theatrical moment",
                  "brief": "One character's raw inner moment, theatrical and confessional, building to a killer final line."},
    "horror": {"label": "👻 Horror Scene", "desc": "Genuinely unsettling",
               "brief": "Genuinely unsettling horror: dread, gore and darkness allowed, no cozy resolutions."},
    "noir": {"label": "🕵️ Noir", "desc": "Hardboiled detective yarn",
             "brief": "Hardboiled detective narration dripping with cynical metaphors, rain and neon."},
    "romance": {"label": "💘 Romance", "desc": "Adult tension, tasteful",
                "brief": "Adult romantic tension, sensual but tasteful, all longing and payoff."},
    "comedy": {"label": "😂 Comedy Sketch", "desc": "Fast jokes, absurdity",
               "brief": "Fast-paced comedy sketch with escalating absurdity and quotable punchlines."},
    "motivational": {"label": "🔥 Motivational", "desc": "High-voltage coach speech",
                     "brief": "High-voltage motivational speech: short hammer sentences, fire, and a call to action."},
    "bedtime": {"label": "🌙 Bedtime", "desc": "Slow, soft, drowsy-safe",
                "brief": "Slow bedtime tale: soft drowsy rhythm, gentle imagery, a sleepy ending, nothing frightening."},
    "epic": {"label": "⚔️ Epic Narration", "desc": "Mythic trailer-scale",
             "brief": "Mythic trailer-scale narration: weighty, dramatic, larger than life."},
}

# ------------------------------------------------- ----- cloud writers ---
# OPTIONAL free-tier cloud backends: 10–100x faster than local CPU inference
# with 70B-class quality, still $0 on their free tiers. Keys are BYO, stored
# chmod-600 in keys.json — values NEVER leave the server (GET only reports
# configured true/false) — and the matching env vars also work.
# Cloud failures fall back to the free local writer instead of erroring.
SCRIPT_CLOUD = {
    "groq": {"label": "Groq ⚡", "key_url": "https://console.groq.com/keys",
             "note": "Free tier, ~2000 tok/s. 70B quality in seconds.",
             "default": "llama-3.3-70b-versatile",
             "models": [{"id": "llama-3.3-70b-versatile", "label": "70B Versatile — best quality"},
                        {"id": "llama-3.1-8b-instant", "label": "8B Instant — fastest"}]},
    "gemini": {"label": "Gemini ✨", "key_url": "https://aistudio.google.com/apikey",
               "note": "Free tier, generous quotas, fast.",
               "default": "gemini-2.0-flash",
               "models": [{"id": "gemini-2.0-flash", "label": "2.0 Flash — fast + sharp"}]},
    "grok": {"label": "Grok 🤖", "key_url": "https://console.x.ai/",
             "note": "xAI models via free credits. Bold, witty voice.",
             "default": "grok-4",
             "models": [{"id": "grok-4", "label": "Grok 4 — flagship"},
                        {"id": "grok-3-mini", "label": "Grok 3 Mini — fast + cheap"}]},
    "claude": {"label": "Claude 🧠", "key_url": "https://console.anthropic.com/",
               "note": "Anthropic models. Best-in-class dialogue and nuance.",
               "default": "claude-sonnet-4-5",
               "models": [{"id": "claude-sonnet-4-5", "label": "Sonnet — best all-rounder"},
                        {"id": "claude-haiku-4-5", "label": "Haiku — fastest"}]},
}
# NOTE (Oct 2026): providers rename models occasionally — if a cloud draft
# starts failing with model-not-found, check their docs and edit the IDs here.
