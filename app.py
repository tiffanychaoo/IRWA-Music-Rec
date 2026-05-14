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
import requests
import spotipy
from flask import Flask, jsonify, redirect, render_template, request, session, url_for
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from spotipy.oauth2 import SpotifyOAuth
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", os.urandom(24))

 
# Config
 
SPOTIFY_CLIENT_ID     = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
SPOTIFY_REDIRECT_URI  = os.getenv("SPOTIFY_REDIRECT_URI", "http://localhost:5000/callback")
LASTFM_API_KEY        = os.getenv("LASTFM_API_KEY")
LASTFM_API_SECRET     = os.getenv("LASTFM_API_SECRET")
LASTFM_USERNAME       = os.getenv("LASTFM_USERNAME", "")
BLUESKY_HANDLE        = os.getenv("BLUESKY_HANDLE", "")
BLUESKY_PASSWORD      = os.getenv("BLUESKY_PASSWORD", "")

SPOTIFY_SCOPE      = "user-top-read user-library-read user-read-recently-played"
SPOTIFY_TIME_RANGES = ["short_term", "medium_term", "long_term"]

# Scoring weights (match notebook)
SCORING_W1  = float(os.getenv("SCORING_W1", "0.4"))
SCORING_W2  = float(os.getenv("SCORING_W2", "0.3"))
SCORING_W3  = float(os.getenv("SCORING_W3", "0.3"))
TASTE_BETA  = float(os.getenv("TASTE_BETA",  "0.45"))
TASTE_GAMMA = float(os.getenv("TASTE_GAMMA", "0.35"))

SOCIAL_SCORE_PEER_OVERLAP_WEIGHT = float(os.getenv("SOCIAL_SCORE_PEER_OVERLAP_WEIGHT", "1.0"))
SOCIAL_SCORE_WEIGHTED_PEER_WEIGHT = float(os.getenv("SOCIAL_SCORE_WEIGHTED_PEER_WEIGHT", "1.0"))

FANS_PER_ARTIST              = int(os.getenv("FANS_PER_ARTIST", "50"))
PEER_GROUP_SEED_ARTIST_COUNT = int(os.getenv("PEER_GROUP_SEED_ARTIST_COUNT", "20"))
MIN_PEER_SUPPORT             = int(os.getenv("MIN_PEER_SUPPORT_FOR_CANDIDATES", "1"))
MAX_PEERS                    = int(os.getenv("MAX_PEERS", "200"))
SECOND_HOP_DISCOUNT          = float(os.getenv("SECOND_HOP_MATCH_DISCOUNT", "0.5"))
CANDIDATE_BUZZ_LIMIT         = 75
POSTS_PER_CANDIDATE          = 30

LASTFM_BASE_URL = "http://ws.audioscrobbler.com/2.0/"
BLUESKY_BASE_URL = "https://bsky.social/xrpc"
CACHE_DIR = Path("./lastfm_cache")
CACHE_DIR.mkdir(exist_ok=True)

URL_PATTERN = re.compile(r"https?://\S+")

# In-memory job store
JOBS: dict[str, dict] = {}

 
# Helpers
 

def normalize_name(value):
    if value is None:
        return None
    n = str(value).strip().lower()
    return n or None

def normalize_tag(tag):
    if not tag:
        return ""
    tag = str(tag).lower().strip().replace('-', ' ')
    return ' '.join(tag.split())

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

def parse_iso_datetime(value):
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None

def iter_nested_items(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        items = []
        for v in value.values():
            items.extend(safe_list(v))
        return items
    return []

 
# Last.fm API (cached)
 

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

def get_similar_artists(artist_name, limit=50):
    payload = lastfm_get("artist.getSimilar", artist=artist_name, limit=limit)
    items = safe_list((payload.get("similarartists") or {}).get("artist"))
    results, seen = [], set()
    for item in items:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        key = normalize_name(name)
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

def get_lastfm_tags(artist_name, limit=10):
    payload = lastfm_get("artist.getTopTags", artist=artist_name)
    items = safe_list((payload.get("toptags") or {}).get("tag"))
    return [
        {"tag": item.get("name"), "weight": int(item.get("count", 0))}
        for item in items[:limit]
        if isinstance(item, dict) and item.get("name")
    ]

def get_lastfm_top_artists(username, limit=100):
    payload = lastfm_get("user.getTopArtists", user=username, limit=limit, period="overall")
    items = safe_list((payload.get("topartists") or {}).get("artist"))
    results = []
    for item in items:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if name:
            results.append({
                "artist_name": str(name).strip(),
                "playcount": int(item.get("playcount", 0)),
            })
    return results

 
# Bluesky
 

def bluesky_auth():
    if not BLUESKY_HANDLE or not BLUESKY_PASSWORD:
        return None
    try:
        r = requests.post(
            f"{BLUESKY_BASE_URL}/com.atproto.server.createSession",
            json={"identifier": BLUESKY_HANDLE, "password": BLUESKY_PASSWORD},
            timeout=30,
        )
        r.raise_for_status()
        return r.json().get("accessJwt")
    except Exception:
        return None

def search_bluesky(query, token, limit=30):
    if not token:
        return []
    posts = []
    cursor = None
    while len(posts) < limit:
        params = {"q": query, "limit": min(100, limit - len(posts)), "sort": "latest"}
        if cursor:
            params["cursor"] = cursor
        try:
            r = requests.get(
                f"{BLUESKY_BASE_URL}/app.bsky.feed.searchPosts",
                headers={"Authorization": f"Bearer {token}"},
                params=params, timeout=30,
            )
            r.raise_for_status()
            data = r.json()
        except Exception:
            break
        for post in data.get("posts", []):
            record = post.get("record") or {}
            text = URL_PATTERN.sub("", record.get("text", "")).strip()
            posts.append({
                "artist_name": query,
                "post_uri": post.get("uri", ""),
                "post_text": text,
                "timestamp": record.get("createdAt"),
                "like_count": post.get("likeCount", 0),
                "repost_count": post.get("repostCount", 0),
            })
            if len(posts) >= limit:
                break
        cursor = data.get("cursor")
        if not cursor or not data.get("posts"):
            break
        time.sleep(0.5)
    return posts

 
# Pipeline
 

def run_pipeline(job_id: str, spotify_token: str):
    job = JOBS[job_id]

    def update(status, progress, message=""):
        job["status"]   = status
        job["progress"] = progress
        job["message"]  = message

    try:
        # ---- Spotify -------------------------------------------------------
        update("running", 5, "Fetching Spotify top artists…")
        sp = spotipy.Spotify(auth=spotify_token)

        spotify_top_artists = {}
        for tr in SPOTIFY_TIME_RANGES:
            resp = sp.current_user_top_artists(limit=50, time_range=tr)
            spotify_top_artists[tr] = [
                {
                    "artist_id": a.get("id"),
                    "name": a.get("name"),
                    "genres": a.get("genres", []),
                    "popularity": a.get("popularity"),
                }
                for a in resp.get("items", [])
            ]
        print(f"Top artists short_term: {[a['name'] for a in spotify_top_artists.get('short_term', [])][:10]}")

        # Flag whether Spotify returned real data
        has_spotify_data = any(
            len(artists) > 0 for artists in spotify_top_artists.values()
        )
        job["has_spotify_data"] = has_spotify_data
        if not has_spotify_data:
            print("WARNING: Spotify returned no data — account likely not on allowlist")

        update("running", 12, "Fetching Spotify top tracks…")
        spotify_top_tracks = {}
        for tr in SPOTIFY_TIME_RANGES:
            resp = sp.current_user_top_tracks(limit=50, time_range=tr)
            spotify_top_tracks[tr] = [
                {
                    "track_id": t.get("id"),
                    "name": t.get("name"),
                    "artist_name": t.get("artists", [{}])[0].get("name"),
                    "duration_ms": t.get("duration_ms"),
                    "popularity": t.get("popularity"),
                }
                for t in resp.get("items", [])
            ]

        update("running", 18, "Fetching saved library…")
        spotify_saved_library = []
        offset = 0
        while True:
            resp = sp.current_user_saved_tracks(limit=50, offset=offset)
            items = resp.get("items", [])
            if not items:
                break
            for item in items:
                track = item.get("track") or {}
                album = track.get("album") or {}
                artists = track.get("artists", [{}])
                spotify_saved_library.append({
                    "track_id": track.get("id"),
                    "name": track.get("name"),
                    "artist_name": artists[0].get("name") if artists else None,
                    "duration_ms": track.get("duration_ms"),
                    "added_at": item.get("added_at"),
                    "album_release_date": album.get("release_date"),
                })
            offset += len(items)
            if len(items) < 50:
                break

        # ---- Last.fm -------------------------------------------------------
        update("running", 25, "Fetching Last.fm top artists…")
        lastfm_top_artists = []
        if LASTFM_USERNAME and LASTFM_API_KEY:
            lastfm_top_artists = get_lastfm_top_artists(LASTFM_USERNAME, limit=100)

        # ---- Known artists set --------------------------------------------
        known_artist_names = set()
        for artist_list in spotify_top_artists.values():
            for a in artist_list:
                key = normalize_name(a.get("name"))
                if key:
                    known_artist_names.add(key)
        for item in lastfm_top_artists:
            key = normalize_name(item.get("artist_name"))
            if key:
                known_artist_names.add(key)
        for item in spotify_saved_library:
            key = normalize_name(item.get("artist_name"))
            if key:
                known_artist_names.add(key)

        # ---- Last.fm tags for known artists --------------------------------
        update("running", 30, "Fetching Last.fm tags for profile…")
        lastfm_artist_tags = {}
        artist_playcounts = {item["artist_name"]: item["playcount"] for item in lastfm_top_artists}

        all_known_names = list({
            a["name"]
            for artist_list in spotify_top_artists.values()
            for a in artist_list
            if a.get("name")
        })[:50]
        for name in all_known_names:
            tags = get_lastfm_tags(name, limit=10)
            if tags:
                lastfm_artist_tags[name] = tags

        # ---- User genre distribution (from Last.fm tags + playcounts) ------
        update("running", 35, "Building user taste profile…")
        tag_counter = Counter()
        for artist_name, tags in lastfm_artist_tags.items():
            artist_weight = artist_playcounts.get(artist_name, 1)
            for tag_info in tags:
                tag_name = normalize_tag(tag_info.get("tag", ""))
                if tag_name:
                    tag_weight = tag_info.get("weight", 0) / 100.0
                    tag_counter[tag_name] += tag_weight * artist_weight

        total_weight = sum(tag_counter.values())
        user_genre_distribution = (
            {tag: weight / total_weight for tag, weight in tag_counter.most_common()}
            if total_weight > 0 else {}
        )

        # ---- Seed artists (Last.fm first, then Spotify) --------------------
        update("running", 40, "Building similar-artist graph…")
        seed_artists = []
        seen_seeds = set()

        all_top_artists = []
        for item in lastfm_top_artists:
            all_top_artists.append(item.get("artist_name"))
        for tr in SPOTIFY_TIME_RANGES:
            for a in spotify_top_artists.get(tr, []):
                all_top_artists.append(a.get("name"))

        for artist_name in all_top_artists:
            key = normalize_name(artist_name)
            if key and key not in seen_seeds:
                seed_artists.append(artist_name)
                seen_seeds.add(key)
            if len(seed_artists) >= PEER_GROUP_SEED_ARTIST_COUNT:
                break

        print(f"Seed artists: {seed_artists[:5]}")

        # ---- Hop 1 ---------------------------------------------------------
        contributors_first = defaultdict(set)
        max_match_first = {}
        display_name = {}

        for seed in seed_artists:
            seed_key = normalize_name(seed)
            if not seed_key:
                continue
            for sim in get_similar_artists(seed, limit=FANS_PER_ARTIST):
                name = sim["artist_name"]
                key = normalize_name(name)
                if not key or key in known_artist_names or key in seen_seeds:
                    continue
                contributors_first[key].add(seed_key)
                mv = sim["match"]
                if key not in max_match_first or mv > max_match_first[key]:
                    max_match_first[key] = mv
                    display_name[key] = name

        update("running", 52, "Expanding second hop…")
        # ---- Hop 2 ---------------------------------------------------------
        anchor_keys = sorted(
            contributors_first.keys(),
            key=lambda k: (-len(contributors_first[k]), -max_match_first.get(k, 0.0)),
        )[:50]

        max_match_second = {}
        for anchor_key in anchor_keys:
            anchor_name = display_name.get(anchor_key)
            if not anchor_name:
                continue
            for sim in get_similar_artists(anchor_name, limit=FANS_PER_ARTIST):
                name = sim["artist_name"]
                key = normalize_name(name)
                if not key or key in known_artist_names or key in seen_seeds or key == anchor_key:
                    continue
                mv = sim["match"] * SECOND_HOP_DISCOUNT
                if key not in max_match_second or mv > max_match_second[key]:
                    max_match_second[key] = mv
                    if key not in display_name:
                        display_name[key] = name

        # ---- Build candidate rows ------------------------------------------
        candidate_rows = []
        total_seeds = len(seed_artists)

        for key, seed_keys in contributors_first.items():
            n = len(seed_keys)
            if n < MIN_PEER_SUPPORT or key in known_artist_names:
                continue
            w = max(
                float(max_match_first.get(key, 0.0)),
                float(max_match_second.get(key, 0.0)),
            )
            candidate_rows.append({
                "artist_name": display_name.get(key, key),
                "candidate_key": key,
                "peers_who_listen": n,
                "peer_overlap_ratio": n / total_seeds if total_seeds else 0.0,
                "weighted_peer_score": w,
                "source": "first_hop",
            })

        for key, mv in max_match_second.items():
            if key in contributors_first or key in known_artist_names:
                continue
            candidate_rows.append({
                "artist_name": display_name.get(key, key),
                "candidate_key": key,
                "peers_who_listen": 0,
                "peer_overlap_ratio": 0.0,
                "weighted_peer_score": float(mv),
                "source": "second_hop",
            })

        # Compute social score
        for row in candidate_rows:
            row["social_score"] = (
                float(row["peer_overlap_ratio"]) ** SOCIAL_SCORE_PEER_OVERLAP_WEIGHT
                * float(row["weighted_peer_score"]) ** SOCIAL_SCORE_WEIGHTED_PEER_WEIGHT
            )

        candidate_rows.sort(key=lambda r: r["social_score"], reverse=True)
        if len(candidate_rows) > MAX_PEERS:
            candidate_rows = candidate_rows[:MAX_PEERS]

        print(f"Candidates after graph expansion: {len(candidate_rows)}")

        # ---- Bluesky buzz --------------------------------------------------
        update("running", 62, "Fetching Bluesky buzz…")
        bsky_token = bluesky_auth()
        top_candidates = [r["artist_name"] for r in candidate_rows[:CANDIDATE_BUZZ_LIMIT]]
        all_posts = []
        for artist_name in top_candidates:
            posts = search_bluesky(artist_name, bsky_token, limit=POSTS_PER_CANDIDATE)
            all_posts.extend(posts)

        # ---- Last.fm tags for candidates -----------------------------------
        update("running", 70, "Fetching candidate tags…")
        existing_tag_keys = set(normalize_name(k) for k in lastfm_artist_tags.keys())
        candidate_names = [r["artist_name"] for r in candidate_rows]
        missing = [n for n in candidate_names if normalize_name(n) not in existing_tag_keys]

        for artist_name in missing[:150]:
            tags = get_lastfm_tags(artist_name, limit=10)
            lastfm_artist_tags[artist_name] = tags

        # ---- Tag text lookup -----------------------------------------------
        tag_text_by_key = {}
        for artist_name, tag_list in lastfm_artist_tags.items():
            artist_key = normalize_name(artist_name)
            tag_names = [normalize_tag(t.get("tag", "")) for t in tag_list if t.get("tag")]
            if artist_key and tag_names:
                tag_text_by_key[artist_key] = " ".join(tag_names)

        # ---- TF-IDF text similarity ----------------------------------------
        update("running", 76, "Computing TF-IDF text similarity…")
        user_documents = {k: tag_text_by_key[k] for k in known_artist_names if tag_text_by_key.get(k)}
        candidate_documents = {
            normalize_name(r["artist_name"]): tag_text_by_key.get(normalize_name(r["artist_name"]), "")
            for r in candidate_rows
        }

        text_similarity_lookup = {k: 0.0 for k in candidate_documents}
        nonempty_user = [(k, d) for k, d in user_documents.items() if d]
        nonempty_candidate = [(k, d) for k, d in candidate_documents.items() if d]

        if nonempty_user and nonempty_candidate:
            corpus = [d for _, d in nonempty_user] + [d for _, d in nonempty_candidate]
            try:
                vectorizer = TfidfVectorizer(stop_words="english", max_features=500)
                tfidf_matrix = vectorizer.fit_transform(corpus)
                user_matrix = tfidf_matrix[:len(nonempty_user)]
                candidate_matrix = tfidf_matrix[len(nonempty_user):]
                user_centroid = np.asarray(user_matrix.mean(axis=0)).reshape(1, -1)
                if np.linalg.norm(user_centroid) > 0:
                    sims = cosine_similarity(candidate_matrix, user_centroid).ravel()
                    for (artist_key, _), sim in zip(nonempty_candidate, sims):
                        text_similarity_lookup[artist_key] = float(sim)
            except Exception as e:
                print(f"TF-IDF failed: {e}")

        # ---- Genre similarity ----------------------------------------------
        genre_similarity_lookup = {}
        for row in candidate_rows:
            artist_key = row["candidate_key"]
            candidate_tags_raw = tag_text_by_key.get(artist_key, "").split()
            if candidate_tags_raw and user_genre_distribution:
                candidate_dist = {normalize_tag(t): 1.0 / len(candidate_tags_raw) for t in candidate_tags_raw}
                all_keys = sorted(set(candidate_dist) | set(user_genre_distribution))
                vec1 = np.array([candidate_dist.get(k, 0) for k in all_keys])
                vec2 = np.array([user_genre_distribution.get(k, 0) for k in all_keys])
                n1, n2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
                if n1 > 0 and n2 > 0:
                    genre_similarity_lookup[artist_key] = float(np.dot(vec1, vec2) / (n1 * n2))

        # ---- Buzz score ----------------------------------------------------
        update("running", 82, "Computing buzz scores…")
        reference_dt = datetime.now(timezone.utc)
        buzz_scores = {}
        posts_by_artist = defaultdict(list)
        for post in all_posts:
            key = normalize_name(post.get("artist_name", ""))
            if key:
                posts_by_artist[key].append(post)

        for row in candidate_rows:
            artist_key = row["candidate_key"]
            artist_posts = posts_by_artist.get(artist_key, [])
            if not artist_posts:
                buzz_scores[artist_key] = 0.0
                continue
            buzz_value = 0.0
            for post in artist_posts:
                engagement = safe_float(post.get("like_count")) + safe_float(post.get("repost_count"))
                post_time = parse_iso_datetime(post.get("timestamp"))
                if post_time:
                    days_ago = (reference_dt - post_time).total_seconds() / 86400
                    decay = float(np.exp(-0.1 * days_ago))
                else:
                    decay = 0.5
                buzz_value += (1 + engagement) * decay
            post_count = len(artist_posts)
            volume_bonus = float(np.log1p(post_count) / np.log1p(50))
            buzz_scores[artist_key] = buzz_value * (1 + 0.3 * volume_bonus)

        max_buzz = max(buzz_scores.values()) if buzz_scores and max(buzz_scores.values()) > 0 else 1.0
        buzz_normalized = {k: v / max_buzz for k, v in buzz_scores.items()}

        # ---- Taste score ---------------------------------------------------
        update("running", 88, "Computing final scores…")
        results = []
        social_scores = [r["social_score"] for r in candidate_rows]
        max_social = max(social_scores) if social_scores and max(social_scores) > 0 else 1.0

        for row in candidate_rows:
            key = row["candidate_key"]
            genre_sim   = genre_similarity_lookup.get(key, 0.0)
            text_sim    = text_similarity_lookup.get(key, 0.0)
            taste_score = (TASTE_BETA * genre_sim + TASTE_GAMMA * text_sim) / (TASTE_BETA + TASTE_GAMMA)
            social_norm = row["social_score"] / max_social
            buzz        = buzz_normalized.get(key, 0.0)
            final       = SCORING_W1 * taste_score + SCORING_W2 * social_norm + SCORING_W3 * buzz

            tags_raw = lastfm_artist_tags.get(row["artist_name"], [])
            tags = [t["tag"] for t in tags_raw[:5] if t.get("tag")]

            results.append({
                "artist_name":   row["artist_name"],
                "final_score":   round(final, 4),
                "taste_score":   round(taste_score, 4),
                "social_score":  round(social_norm, 4),
                "buzz_score":    round(buzz, 4),
                "peers_who_listen": row["peers_who_listen"],
                "tags":          tags,
            })

        results.sort(key=lambda x: x["final_score"], reverse=True)

        # ---- Ablation ------------------------------------------------------
        ablation = []
        configs = [
            {"label": "Taste only",     "w1": 1.0, "w2": 0.0, "w3": 0.0},
            {"label": "Taste + Social", "w1": 0.6, "w2": 0.4, "w3": 0.0},
            {"label": "Taste + Buzz",   "w1": 0.6, "w2": 0.0, "w3": 0.4},
            {"label": "All signals",    "w1": 0.4, "w2": 0.3, "w3": 0.3},
        ]
        for cfg in configs:
            scored = sorted(
                results,
                key=lambda r: cfg["w1"] * r["taste_score"]
                            + cfg["w2"] * r["social_score"]
                            + cfg["w3"] * r["buzz_score"],
                reverse=True,
            )
            ablation.append({"label": cfg["label"], "top10": [r["artist_name"] for r in scored[:10]]})

        job["result"] = {
            "recommendations":   results[:50],
            "ablation":          ablation,
            "genre_distribution": dict(list(user_genre_distribution.items())[:15]),
            "tag_distribution":   dict(list(user_genre_distribution.items())[:15]),
            "seed_artists":       seed_artists,
            "total_candidates":   len(results),
        }
        update("done", 100, "Complete")

    except Exception as exc:
        import traceback
        print(traceback.format_exc())
        job["status"] = "error"
        job["error"]  = str(exc)

 
# Spotify OAuth
 

def make_spotify_oauth():
    return SpotifyOAuth(
        client_id=SPOTIFY_CLIENT_ID,
        client_secret=SPOTIFY_CLIENT_SECRET,
        redirect_uri=SPOTIFY_REDIRECT_URI,
        scope=SPOTIFY_SCOPE,
        cache_path=None,
        open_browser=False,
    )

 
# Routes
 

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/login")
def login():
    oauth = make_spotify_oauth()
    return redirect(oauth.get_authorize_url())

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
    lastfm_user = LASTFM_USERNAME or "not configured"
    lastfm_live = False
    return render_template("loading.html", job_id=job_id, lastfm_user=lastfm_user, lastfm_live=lastfm_live)

@app.route("/status/<job_id>")
def status(job_id):
    job = JOBS.get(job_id)
    if not job:
        return jsonify({"status": "error", "error": "Job not found"}), 404
    return jsonify({"status": job["status"], "progress": job["progress"], "message": job["message"], "error": job.get("error")})

@app.route("/spotify-status/<job_id>")
def spotify_status(job_id):
    job = JOBS.get(job_id)
    if not job:
        return jsonify({"has_spotify_data": False})
    return jsonify({"has_spotify_data": job.get("has_spotify_data", None)})

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
