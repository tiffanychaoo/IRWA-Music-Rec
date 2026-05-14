import hashlib
import os
import json
import re
import time
import threading
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pylast
import requests
import spotipy
from flask import (Flask, jsonify, redirect, render_template,
                   request, session, url_for)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from spotipy.oauth2 import SpotifyOAuth
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", os.urandom(24))

# CONFIG
SPOTIFY_CLIENT_ID     = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
SPOTIFY_REDIRECT_URI  = os.getenv("SPOTIFY_REDIRECT_URI", "http://localhost:5000/callback")
LASTFM_API_KEY        = os.getenv("LASTFM_API_KEY")
LASTFM_API_SECRET     = os.getenv("LASTFM_API_SECRET")
LASTFM_USERNAME       = os.getenv("LASTFM_USERNAME")
BLUESKY_HANDLE        = os.getenv("BLUESKY_HANDLE", "")
BLUESKY_PASSWORD      = os.getenv("BLUESKY_PASSWORD", "")

SPOTIFY_SCOPE = "user-top-read user-library-read user-read-recently-played"
TIME_RANGES   = ["short_term", "medium_term", "long_term"]

SCORING_W1  = float(os.getenv("SCORING_W1", "0.4"))
SCORING_W2  = float(os.getenv("SCORING_W2", "0.3"))
SCORING_W3  = float(os.getenv("SCORING_W3", "0.3"))
TASTE_BETA    = float(os.getenv("TASTE_BETA",    "0.45"))   # Genre
TASTE_GAMMA   = float(os.getenv("TASTE_GAMMA",   "0.35"))   # Text/tags
TASTE_DELTA   = float(os.getenv("TASTE_DELTA",   "0.1"))    # Duration
TASTE_EPSILON = float(os.getenv("TASTE_EPSILON", "0.1"))    # Recency

SOCIAL_SCORE_PEER_OVERLAP_WEIGHT = float(os.getenv("SOCIAL_SCORE_PEER_OVERLAP_WEIGHT", "1.0"))
SOCIAL_SCORE_WEIGHTED_PEER_WEIGHT = float(os.getenv("SOCIAL_SCORE_WEIGHTED_PEER_WEIGHT", "1.0"))

FANS_PER_ARTIST              = int(os.getenv("FANS_PER_ARTIST", "10"))
PEER_GROUP_SEED_ARTIST_COUNT = int(os.getenv("PEER_GROUP_SEED_ARTIST_COUNT", "5"))
MIN_PEER_SUPPORT             = int(os.getenv("MIN_PEER_SUPPORT_FOR_CANDIDATES", "2"))
MAX_PEERS                    = int(os.getenv("MAX_PEERS", "100"))
SECOND_HOP_DISCOUNT          = float(os.getenv("SECOND_HOP_MATCH_DISCOUNT", "0.5"))
BLUESKY_POST_LIMIT           = int(os.getenv("BLUESKY_POST_LIMIT", "500"))
BLUESKY_PAGE_SIZE            = 100
SHARED_ARTIST_QUERY_LIMIT    = int(os.getenv("SHARED_ARTIST_QUERY_LIMIT", "10"))
CANDIDATE_BUZZ_LIMIT         = 75
POSTS_PER_CANDIDATE          = 30

LASTFM_BASE_URL     = "http://ws.audioscrobbler.com/2.0/"
BLUESKY_BASE_URL    = "https://bsky.social/xrpc"
CACHE_DIR           = Path("./lastfm_cache")
CACHE_DIR.mkdir(exist_ok=True)

JOBS: dict[str, dict] = {}

# HELPERS
def normalize_name(value):
    if value is None:
        return None
    n = str(value).strip().lower()
    return n or None

def safe_float(value, default=0.0):
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

def safe_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]

def chunked(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]

def normalize_score_lookup(lookup, allow_negative=False):
    if not lookup:
        return {}
    import pandas as pd
    series = pd.Series(lookup, dtype=float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    mn, mx = float(series.min()), float(series.max())
    if allow_negative and mx > mn:
        return {k: float((v - mn) / (mx - mn)) for k, v in series.items()}
    if mx > 0:
        return {k: float(v / mx) for k, v in series.items()}
    return {k: 0.0 for k in series.index}

def distribution_cosine_similarity(left, right):
    left = left or {}
    right = right or {}
    if not left or not right:
        return None
    keys = sorted(set(left) | set(right))
    lv = np.array([safe_float(left.get(k, 0.0)) for k in keys])
    rv = np.array([safe_float(right.get(k, 0.0)) for k in keys])
    ln, rn = np.linalg.norm(lv), np.linalg.norm(rv)
    if ln == 0 or rn == 0:
        return None
    return float(cosine_similarity(lv.reshape(1, -1), rv.reshape(1, -1))[0, 0])

def exp_decay(dt_value, lambda_per_day=0.05):
    if dt_value is None:
        return 1.0
    if isinstance(dt_value, str):
        try:
            dt_value = datetime.fromisoformat(dt_value.replace("Z", "+00:00"))
        except Exception:
            return 1.0
    now = datetime.now(timezone.utc)
    if dt_value.tzinfo is None:
        dt_value = dt_value.replace(tzinfo=timezone.utc)
    days = max((now - dt_value).total_seconds(), 0.0) / 86400.0
    return float(np.exp(-lambda_per_day * days))

# LAST.FM API (cached)

def lastfm_get(method, **params):
    key_data = json.dumps({"method": method, "params": {k: str(v) for k, v in sorted(params.items())}}, sort_keys=True)
    digest = hashlib.sha1(key_data.encode()).hexdigest()[:12]
    cache_path = CACHE_DIR / f"{method.replace('.','_')}_{digest}.json"
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text())
        except Exception:
            pass
    req_params = {"method": method, "api_key": LASTFM_API_KEY, "format": "json", **params}
    try:
        r = requests.get(LASTFM_BASE_URL, params=req_params, timeout=30)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, dict) and data.get("error"):
            return {}
        cache_path.write_text(json.dumps(data))
        return data
    except Exception:
        return {}
    finally:
        time.sleep(0.25)

def get_similar_artists(artist_name, limit=FANS_PER_ARTIST):
    payload = lastfm_get("artist.getSimilar", artist=artist_name, limit=limit)
    items = safe_list((payload.get("similarartists") or {}).get("artist"))
    results, seen = [], set()
    for item in items:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        key  = normalize_name(name)
        if not key or key in seen:
            continue
        seen.add(key)
        try:
            match = float(item.get("match", 0))
            if match > 1.0:
                match /= 100.0
            match = float(np.clip(match, 0.0, 1.0))
        except (TypeError, ValueError):
            match = 0.0
        results.append({"artist_name": str(name).strip(), "match": match})
    return results

def get_lastfm_tags(artist_name):
    payload = lastfm_get("artist.getTopTags", artist=artist_name)
    items = safe_list((payload.get("toptags") or {}).get("tag"))
    return [
        {"tag": item.get("name"), "weight": int(item.get("count", 0))}
        for item in items[:10]
        if isinstance(item, dict) and item.get("name")
    ]

# BLUESKYAPI

bluesky_access_token = None

def bluesky_auth():
    global bluesky_access_token
    if not BLUESKY_HANDLE or not BLUESKY_PASSWORD:
        return None
    try:
        r = requests.post(
            f"{BLUESKY_BASE_URL}/com.atproto.server.createSession",
            json={"identifier": BLUESKY_HANDLE, "password": BLUESKY_PASSWORD},
            timeout=30,
        )
        r.raise_for_status()
        token = r.json().get("accessJwt")
        bluesky_access_token = token
        return token
    except Exception:
        return None

def search_bluesky(query, token, limit=None):
    global bluesky_access_token
    if not token:
        return []
    if limit is None:
        limit = BLUESKY_POST_LIMIT
    limit = max(1, int(limit))
    posts = []
    cursor = None
    while len(posts) < limit:
        page_limit = min(BLUESKY_PAGE_SIZE, limit - len(posts))
        params = {"q": query, "limit": page_limit, "sort": "latest"}
        if cursor:
            params["cursor"] = cursor
        headers = {"Authorization": f"Bearer {token}"}
        try:
            r = requests.get(
                f"{BLUESKY_BASE_URL}/app.bsky.feed.searchPosts",
                headers=headers,
                params=params, timeout=30,
            )
            if r.status_code == 401:
                token = bluesky_auth()
                bluesky_access_token = token
                headers = {"Authorization": f"Bearer {token}"}
                r = requests.get(
                    f"{BLUESKY_BASE_URL}/app.bsky.feed.searchPosts",
                    headers=headers,
                    params=params, timeout=30,
                )
            r.raise_for_status()
            data = r.json()
        except Exception:
            break
        for post in data.get("posts", []):
            record = post.get("record") or {}
            raw_text = record.get("text", "")
            text = re.sub(r"https?://\S+", "", raw_text).strip()
            posts.append({
                "text": text,
                "raw_text": raw_text,
                "like_count": post.get("likeCount", 0),
                "repost_count": post.get("repostCount", 0),
                "timestamp": record.get("createdAt"),
            })
            if len(posts) >= limit:
                break
        cursor = data.get("cursor")
        if not cursor or not data.get("posts"):
            break
        time.sleep(0.5)
    return posts

# RUN PIPELINE
def run_pipeline(job_id: str, spotify_token: str):
    job = JOBS[job_id]

    def update(status, progress, message=""):
        job["status"]   = status
        job["progress"] = progress
        job["message"]  = message

    try:
        #  Spotify 
        update("running", 5, "Fetching Spotify top artists…")
        sp = spotipy.Spotify(auth=spotify_token)

        top_artists = {}
        for tr in TIME_RANGES:
            resp = sp.current_user_top_artists(limit=50, time_range=tr)
            top_artists[tr] = [
                {
                    "artist_id": a.get("id"),
                    "name": a.get("name"),
                    "genres": a.get("genres", []),
                    "popularity": a.get("popularity"),
                }
                for a in resp.get("items", [])
            ]

        update("running", 15, "Fetching Spotify top tracks…")
        top_tracks = {}
        for tr in TIME_RANGES:
            resp = sp.current_user_top_tracks(limit=50, time_range=tr)
            top_tracks[tr] = [
                {
                    "track_id": t.get("id"),
                    "name": t.get("name"),
                    "artist_ids": [a.get("id") for a in t.get("artists", [])],
                    "popularity": t.get("popularity"),
                    "duration_ms": t.get("duration_ms"),
                }
                for t in resp.get("items", [])
            ]

        update("running", 20, "Fetching saved library…")
        saved_library = []
        offset = 0
        while True:
            resp = sp.current_user_saved_tracks(limit=50, offset=offset)
            items = resp.get("items", [])
            if not items:
                break
            for item in items:
                track = item.get("track") or {}
                album = track.get("album") or {}
                saved_library.append({
                    "track_id": track.get("id"),
                    "name": track.get("name"),
                    "artist_ids": [a.get("id") for a in track.get("artists", [])],
                    "added_at": item.get("added_at"),
                    "album_release_date": album.get("release_date"),
                })
            offset += len(items)
            if len(items) < 50:
                break

        # Genre/tag distributions
        update("running", 30, "Building user profile…")
        genre_counter = Counter()
        unique_artists = {}
        for artist_list in top_artists.values():
            for a in artist_list:
                key = a.get("artist_id") or a.get("name")
                if key and key not in unique_artists:
                    unique_artists[key] = a
        for a in unique_artists.values():
            genre_counter.update(a.get("genres", []))
        total_genre = sum(genre_counter.values())
        genre_distribution = (
            {g: c / total_genre for g, c in genre_counter.most_common()}
            if total_genre else {}
        )

        # Last.fm tags for known artists
        update("running", 35, "Fetching Last.fm tags…")
        all_artist_names = list({
            a["name"]
            for artist_list in top_artists.values()
            for a in artist_list
            if a.get("name")
        })[:50]
        artist_tags: dict[str, list] = {}
        for name in all_artist_names:
            artist_tags[name] = get_lastfm_tags(name)

        tag_weights: dict[str, float] = defaultdict(float)
        for tag_list in artist_tags.values():
            for entry in tag_list:
                if entry.get("tag"):
                    tag_weights[entry["tag"]] += float(entry.get("weight", 0))
        total_tag = sum(tag_weights.values())
        tag_distribution = (
            {t: w / total_tag for t, w in sorted(tag_weights.items(), key=lambda x: x[1], reverse=True)}
            if total_tag else {}
        )

        # Known library set
        known_artist_names: set[str] = set()
        for artist_list in top_artists.values():
            for a in artist_list:
                key = normalize_name(a.get("name"))
                if key:
                    known_artist_names.add(key)

        # Pipeline 2: similar artist candidates
        update("running", 40, "Discovering similar artists…")
        seed_artists = []
        seen_seeds: set[str] = set()
        for name in all_artist_names:
            key = normalize_name(name)
            if key and key not in seen_seeds:
                seed_artists.append(name)
                seen_seeds.add(key)
            if len(seed_artists) >= PEER_GROUP_SEED_ARTIST_COUNT:
                break

        contributors_first: dict[str, set] = defaultdict(set)
        max_match_first:    dict[str, float] = {}
        display_name:       dict[str, str]   = {}
        seed_norm_set = {normalize_name(s) for s in seed_artists if normalize_name(s)}

        for seed in seed_artists:
            seed_key = normalize_name(seed)
            if not seed_key:
                continue
            for sim in get_similar_artists(seed, limit=FANS_PER_ARTIST):
                name = sim["artist_name"]
                key  = normalize_name(name)
                if not key or key in known_artist_names or key in seed_norm_set:
                    continue
                contributors_first[key].add(seed_key)
                mv = sim["match"]
                if key not in max_match_first or mv > max_match_first[key]:
                    max_match_first[key] = mv
                    display_name[key] = name

        update("running", 50, "Expanding second hop…")
        anchor_keys = sorted(
            contributors_first.keys(),
            key=lambda k: (-len(contributors_first[k]), -max_match_first.get(k, 0.0)),
        )[:MAX_PEERS]
        max_match_second: dict[str, float] = {}
        for anchor_key in anchor_keys:
            anchor_name = display_name.get(anchor_key)
            if not anchor_name:
                continue
            for sim in get_similar_artists(anchor_name, limit=FANS_PER_ARTIST):
                name = sim["artist_name"]
                key  = normalize_name(name)
                if not key or key in known_artist_names or key in seed_norm_set or key == anchor_key:
                    continue
                mv = sim["match"] * SECOND_HOP_DISCOUNT
                if key not in max_match_second or mv > max_match_second[key]:
                    max_match_second[key] = mv
                    if key not in display_name:
                        display_name[key] = name

        candidate_rows = []
        total_seeds = len(seed_artists)
        for key, seed_keys in contributors_first.items():
            n = len(seed_keys)
            if n < MIN_PEER_SUPPORT or key in known_artist_names:
                continue
            w = max(max_match_first.get(key, 0.0), max_match_second.get(key, 0.0))
            candidate_rows.append({
                "artist_name": display_name.get(key, key),
                "normalized_artist_name": key,
                "peers_who_listen": n,
                "peer_overlap_ratio": n / total_seeds if total_seeds else 0.0,
                "weighted_peer_score": w,
                "social_score": (n / total_seeds if total_seeds else 0.0) * w,
            })
        for key, mv in max_match_second.items():
            if key in contributors_first or key in known_artist_names:
                continue
            candidate_rows.append({
                "artist_name": display_name.get(key, key),
                "normalized_artist_name": key,
                "peers_who_listen": 0,
                "peer_overlap_ratio": 0.0,
                "weighted_peer_score": mv,
                "social_score": 0.0,
            })

        # Bluesky buzz
        update("running", 60, "Fetching Bluesky buzz…")
        bsky_token = bluesky_auth()
        top_candidates = sorted(candidate_rows, key=lambda r: r["social_score"], reverse=True)
        candidate_names = [r["artist_name"] for r in top_candidates[:CANDIDATE_BUZZ_LIMIT]]
        bluesky_posts: dict[str, list] = {}
        for name in candidate_names:
            try:
                posts = search_bluesky(name, bsky_token, limit=POSTS_PER_CANDIDATE)
                if posts:
                    bluesky_posts[normalize_name(name)] = posts
            except Exception:
                pass

        analyzer = None
        buzz_raw: dict[str, float] = {}
        for key, posts in bluesky_posts.items():
            reference_dt = datetime.now(timezone.utc)
            buzz_value = 0.0
            for p in posts:
                engagement = safe_float(p.get("like_count")) + safe_float(p.get("repost_count"))
                post_time = p.get("timestamp")
                decay = exp_decay(post_time, lambda_per_day=0.1) if post_time else 0.5
                buzz_value += (1 + engagement) * decay
            post_count = len(posts)
            volume_bonus = np.log1p(post_count) / np.log1p(50)
            buzz_raw[key] = buzz_value * (1 + 0.3 * volume_bonus)
        norm_buzz = normalize_score_lookup(buzz_raw)

        # Last.fm tags for candidates
        update("running", 68, "Fetching candidate tags…")
        candidate_tag_lookup: dict[str, list] = {}
        for row in candidate_rows[:50]:
            name = row["artist_name"]
            key  = row["normalized_artist_name"]
            tags = get_lastfm_tags(name)
            if tags:
                candidate_tag_lookup[key] = [t["tag"] for t in tags]
                # Also cache weight-based distribution for similarity
                tw = {t["tag"]: float(t["weight"]) for t in tags if t.get("tag")}
                tot = sum(tw.values())
                row["tag_distribution"] = {t: w / tot for t, w in tw.items()} if tot else {}
            else:
                candidate_tag_lookup[key] = []
                row["tag_distribution"] = {}

        # Taste score
        update("running", 75, "Computing taste scores…")
        # Artist genre lookup from top_artists
        artist_genre_lookup: dict[str, list] = {}
        for artist_list in top_artists.values():
            for a in artist_list:
                key = normalize_name(a.get("name"))
                if key and a.get("genres") and key not in artist_genre_lookup:
                    artist_genre_lookup[key] = [g.lower() for g in a["genres"]]

        # Duration profile
        durations = []
        for track_list in top_tracks.values():
            for t in track_list:
                if t.get("duration_ms"):
                    try:
                        durations.append(float(t["duration_ms"]))
                    except (TypeError, ValueError):
                        pass
        duration_centroid = float(np.mean(durations)) if durations else None
        duration_std = float(np.std(durations, ddof=0)) if durations else None

        # Release year profile
        years = []
        for item in saved_library:
            date_str = item.get("album_release_date")
            if date_str:
                for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
                    try:
                        years.append(int(datetime.strptime(date_str, fmt).year))
                        break
                    except ValueError:
                        continue
        release_year_mean = float(np.mean(years)) if years else None
        release_year_std = float(np.std(years, ddof=1)) if len(years) > 1 else 0.0

        taste_raw: dict[str, float] = {}
        for row in candidate_rows:
            key = row["normalized_artist_name"]
            candidate_genres = artist_genre_lookup.get(key, [])
            genre_sim = None
            if candidate_genres and genre_distribution:
                cg_dist = {g: 1.0 / len(candidate_genres) for g in candidate_genres}
                genre_sim = distribution_cosine_similarity(cg_dist, genre_distribution)

            tag_dist = row.get("tag_distribution", {})
            tag_sim = None
            if tag_dist and tag_distribution:
                tag_sim = distribution_cosine_similarity(tag_dist, tag_distribution)

            # Duration similarity
            dur_sim = None
            if duration_centroid is not None and duration_std and duration_std > 0:
                candidate_dur = row.get("duration_ms")
                if candidate_dur:
                    dur_sim = float(np.exp(-0.5 * ((candidate_dur - duration_centroid) / duration_std) ** 2))

            # Recency similarity
            rec_sim = None
            if release_year_mean is not None and release_year_std and release_year_std > 0:
                candidate_year = row.get("release_year")
                if candidate_year:
                    rec_sim = float(np.exp(-0.5 * ((candidate_year - release_year_mean) / release_year_std) ** 2))

            components = {k: v for k, v in {
                "genre": genre_sim, "tag": tag_sim, "duration": dur_sim, "recency": rec_sim
            }.items() if v is not None}
            weights = {"genre": TASTE_BETA, "tag": TASTE_GAMMA, "duration": TASTE_DELTA, "recency": TASTE_EPSILON}
            if not components:
                taste_raw[key] = 0.0
            else:
                wt = sum(weights[k] for k in components)
                taste_raw[key] = sum((weights[k] / wt) * components[k] for k in components)
        norm_taste = normalize_score_lookup(taste_raw, allow_negative=True)

        # Social score
        social_raw = {}
        for r in candidate_rows:
            key = r["normalized_artist_name"]
            social_raw[key] = (
                float(r["peer_overlap_ratio"]) ** SOCIAL_SCORE_PEER_OVERLAP_WEIGHT
            ) * (
                float(r["weighted_peer_score"]) ** SOCIAL_SCORE_WEIGHTED_PEER_WEIGHT
            )
        norm_social = normalize_score_lookup(social_raw)

        # Buzz score
        buzz_score: dict[str, float] = {}
        for row in candidate_rows:
            key = row["normalized_artist_name"]
            buzz_score[key] = norm_buzz.get(key, 0.0)
        norm_buzz_final = normalize_score_lookup(buzz_score)

        # Final score 
        update("running", 82, "Computing final scores…")
        results = []
        for row in candidate_rows:
            key = row["normalized_artist_name"]
            ts  = safe_float(norm_taste.get(key, 0.0))
            ss  = safe_float(norm_social.get(key, 0.0))
            bs  = safe_float(norm_buzz_final.get(key, 0.0))
            final = SCORING_W1 * ts + SCORING_W2 * ss + SCORING_W3 * bs
            results.append({
                "artist_name": row["artist_name"],
                "final_score": round(final, 4),
                "taste_score": round(ts, 4),
                "social_score": round(ss, 4),
                "buzz_score": round(bs, 4),
                "peers_who_listen": row["peers_who_listen"],
                "tags": candidate_tag_lookup.get(key, [])[:5],
            })

        results.sort(key=lambda x: x["final_score"], reverse=True)

        update("running", 90, "Computing ablation…")

        # Ablation 
        ablation = []
        configs = [
            {"label": "Taste only",     "w1": 1.0, "w2": 0.0,  "w3": 0.0},
            {"label": "Social only",    "w1": 0.0, "w2": 1.0,  "w3": 0.0},
            {"label": "Buzz only",      "w1": 0.0, "w2": 0.0,  "w3": 1.0},
            {"label": "All signals",    "w1": 0.4, "w2": 0.3,  "w3": 0.3},
            {"label": "Buzz-Forward",   "w1": 0.2, "w2": 0.2,  "w3": 0.6},
        ]
        for cfg in configs:
            scored = sorted(
                results,
                key=lambda r: cfg["w1"] * r["taste_score"]
                            + cfg["w2"] * r["social_score"]
                            + cfg["w3"] * r["buzz_score"],
                reverse=True,
            )
            ablation.append({
                "label": cfg["label"],
                "top10": [r["artist_name"] for r in scored[:10]],
            })

        job["result"] = {
            "recommendations": results[:50],
            "ablation": ablation,
            "genre_distribution": dict(list(genre_distribution.items())[:15]),
            "tag_distribution": dict(list(tag_distribution.items())[:15]),
            "seed_artists": seed_artists,
            "total_candidates": len(results),
        }
        update("done", 100, "Complete")

    except Exception as exc:
        job["status"] = "error"
        job["error"]  = str(exc)

# Spotify OAUTH helpers

def make_spotify_oauth():
    return SpotifyOAuth(
        client_id=SPOTIFY_CLIENT_ID,
        client_secret=SPOTIFY_CLIENT_SECRET,
        redirect_uri=SPOTIFY_REDIRECT_URI,
        scope=SPOTIFY_SCOPE,
        cache_path=None,
        open_browser=False,
    )

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/login")
def login():
    oauth = make_spotify_oauth()
    auth_url = oauth.get_authorize_url()
    return redirect(auth_url)

@app.route("/callback")
def callback():
    code = request.args.get("code")
    if not code:
        return redirect(url_for("index"))
    oauth = make_spotify_oauth()
    token_info = oauth.get_access_token(code, as_dict=True)
    access_token = token_info.get("access_token")
    if not access_token:
        return redirect(url_for("index"))

    job_id = str(uuid.uuid4())
    JOBS[job_id] = {"status": "running", "progress": 0, "message": "Starting…", "result": None, "error": None}
    thread = threading.Thread(target=run_pipeline, args=(job_id, access_token), daemon=True)
    thread.start()
    return redirect(url_for("loading", job_id=job_id))

@app.route("/loading/<job_id>")
def loading(job_id):
    return render_template("loading.html", job_id=job_id)

@app.route("/status/<job_id>")
def status(job_id):
    job = JOBS.get(job_id)
    if not job:
        return jsonify({"status": "error", "error": "Job not found"}), 404
    return jsonify({
        "status":   job["status"],
        "progress": job["progress"],
        "message":  job["message"],
        "error":    job.get("error"),
    })

@app.route("/results/<job_id>")
def results(job_id):
    job = JOBS.get(job_id)
    if not job or job["status"] != "done":
        return redirect(url_for("loading", job_id=job_id))
    return render_template("results.html", job_id=job_id, data=job["result"])

@app.route("/api/results/<job_id>")
def api_results(job_id):
    job = JOBS.get(job_id)
    if not job:
        return jsonify({"error": "not found"}), 404
    return jsonify(job.get("result", {}))

if __name__ == "__main__":
    app.run(debug=True)
