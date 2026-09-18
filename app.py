import csv
import json
import sqlite3
from pathlib import Path
from urllib.parse import quote_plus, urlencode
from urllib.request import urlopen

from flask import Flask, g, jsonify, redirect, render_template, request, url_for
from transformers import pipeline

# AI assistance disclosure:
# ChatGPT helped explain Flask, SQLite, CSV reading,
# and Hugging Face pipeline usage. I reviewed and typed this code
# as part of my CS50 final project.

app = Flask(__name__)

BASE_FOLDER = Path(__file__).parent
DATA_FILE = BASE_FOLDER / "data" / "songs.csv"
DATABASE_FILE = BASE_FOLDER / "moodtune.db"

emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base"
)

EMOTION_MAPPING = {
    "Joy": "Happy",
    "Sadness": "Sad",
    "Anger": "Anger",
    "Disgust": "Disgust",
    "Fear": "Fear",
    "Surprise": "Surprise",
    "Neutral": "Neutral"
}


def choose_music_emotion(detected_emotion, listening_goal):
    """Choose the emotion used for music recommendations."""

    detected_csv_emotion = EMOTION_MAPPING.get(detected_emotion, "Neutral")

    if listening_goal == "boost":
        return "Happy"

    if listening_goal == "calm":
        return "Neutral"

    return detected_csv_emotion


def get_db():
    """Open one SQLite database connection for the current request."""
    if "db" not in g:
        g.db = sqlite3.connect(DATABASE_FILE)
        g.db.row_factory = sqlite3.Row

    return g.db


@app.teardown_appcontext
def close_db(error=None):
    """Close the database after each request."""
    db = g.pop("db", None)

    if db is not None:
        db.close()


def initialize_database():
    """Create the analysis-history and favorites tables if they do not exist."""
    db = sqlite3.connect(DATABASE_FILE)

    db.execute("""
        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            mood_text TEXT NOT NULL,
            emotion TEXT NOT NULL,
            confidence REAL NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    db.execute("""
        CREATE TABLE IF NOT EXISTS favorites (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            artist TEXT NOT NULL,
            emotion TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    db.execute("""
        CREATE TABLE IF NOT EXISTS song_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            artist TEXT NOT NULL,
            feedback TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    db.commit()
    db.close()


preview_cache = {}


def get_feedback_scores():
    """Calculate Likes and Not for me scores for every song."""

    # AI assistance disclosure:
    # ChatGPT helped explain this SQLite aggregation query.
    # I reviewed and typed it for my CS50 final project.

    db = get_db()

    rows = db.execute("""
        SELECT
            title,
            artist,
            SUM(CASE WHEN feedback = 'like' THEN 1 ELSE 0 END) AS likes,
            SUM(CASE WHEN feedback = 'dislike' THEN 1 ELSE 0 END) AS dislikes
        FROM song_feedback
        GROUP BY title, artist
    """).fetchall()

    scores = {}

    for row in rows:
        scores[(row["title"], row["artist"])] = {
            "likes": row["likes"],
            "dislikes": row["dislikes"]
        }

    return scores


def get_itunes_preview(title, artist):
    """Get an Apple Music/iTunes 30-second preview for one song."""
    cache_key = f"{title} {artist}"

    if cache_key in preview_cache:
        return preview_cache[cache_key]

    parameters = urlencode({
        "term": cache_key,
        "media": "music",
        "entity": "song",
        "limit": 1,
        "country": "us"
    })

    url = f"https://itunes.apple.com/search?{parameters}"

    try:
        with urlopen(url, timeout=8) as response:
            data = json.load(response)

        results = data.get("results", [])

        if results:
            result = results[0]

            preview = {
                "preview_url": result.get("previewUrl"),
                "itunes_url": result.get("trackViewUrl")
            }
        else:
            preview = {
                "preview_url": None,
                "itunes_url": None
            }

    except (OSError, ValueError):
        preview = {
            "preview_url": None,
            "itunes_url": None
        }

    preview_cache[cache_key] = preview
    return preview


def get_recommendations(recommended_emotion):
    """Return and rank songs using the user's saved feedback."""
    csv_emotion = recommended_emotion
    recommendations = []
    feedback_scores = get_feedback_scores()

    with open(DATA_FILE, newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for song in reader:
            if song["emotion"] == csv_emotion:
                song["spotify_url"] = (
                    "https://open.spotify.com/search/"
                    + quote_plus(f"{song['title']} {song['artist']}")
                )

                preview = get_itunes_preview(song["title"], song["artist"])
                song["preview_url"] = preview["preview_url"]
                song["itunes_url"] = preview["itunes_url"]

                score = feedback_scores.get(
                    (song["title"], song["artist"]),
                    {"likes": 0, "dislikes": 0}
                )

                song["likes"] = score["likes"]
                song["dislikes"] = score["dislikes"]
                song["preference_score"] = score["likes"] - score["dislikes"]

                recommendations.append(song)

    recommendations.sort(
        key=lambda song: (
            song["preference_score"],
            song["likes"]
        ),
        reverse=True
    )

    return recommendations


def get_history_preview_song(detected_emotion):
    """Return one ranked song for a saved mood-history record."""

    csv_emotion = EMOTION_MAPPING.get(detected_emotion, "Neutral")
    feedback_scores = get_feedback_scores()
    candidates = []

    with open(DATA_FILE, newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            if row["emotion"] == csv_emotion:
                song = dict(row)

                score = feedback_scores.get(
                    (song["title"], song["artist"]),
                    {"likes": 0, "dislikes": 0}
                )

                song["preference_score"] = (
                    score["likes"] - score["dislikes"]
                )

                candidates.append(song)

    candidates.sort(
        key=lambda song: song["preference_score"],
        reverse=True
    )

    if not candidates:
        return None

    song = candidates[0]

    song["spotify_url"] = (
        "https://open.spotify.com/search/"
        + quote_plus(f"{song['title']} {song['artist']}")
    )

    preview = get_itunes_preview(song["title"], song["artist"])
    song["preview_url"] = preview["preview_url"]
    song["itunes_url"] = preview["itunes_url"]

    return song


@app.route("/", methods=["GET", "POST"])
def index():
    mood_text = None
    emotion = None
    confidence = None
    listening_goal = "match"
    recommended_emotion = None
    recommendations = []

    if request.method == "POST":
        mood_text = request.form.get("mood_text", "").strip()
        listening_goal = request.form.get("listening_goal", "match")

        if mood_text:
            prediction = emotion_classifier(mood_text)[0]
            emotion = prediction["label"].capitalize()
            confidence = round(prediction["score"] * 100, 2)
            recommended_emotion = choose_music_emotion(emotion, listening_goal)
            recommendations = get_recommendations(recommended_emotion)

            db = get_db()
            db.execute(
                "INSERT INTO analyses (mood_text, emotion, confidence) VALUES (?, ?, ?)",
                (mood_text, emotion, confidence)
            )
            db.commit()

    return render_template(
        "index.html",
        mood_text=mood_text,
        emotion=emotion,
        confidence=confidence,
        listening_goal=listening_goal,
        recommended_emotion=recommended_emotion,
        recommendations=recommendations
    )


@app.route("/history")
def history():
    db = get_db()

    analyses = db.execute("""
        SELECT mood_text, emotion, confidence, created_at
        FROM analyses
        ORDER BY created_at DESC
        LIMIT 10
    """).fetchall()

    songs_by_emotion = {}
    analyses_with_music = []

    for analysis in analyses:
        item = dict(analysis)
        emotion = item["emotion"]

        if emotion not in songs_by_emotion:
            songs_by_emotion[emotion] = get_history_preview_song(emotion)

        item["replay_song"] = songs_by_emotion[emotion]
        analyses_with_music.append(item)

    return render_template(
        "history.html",
        analyses=analyses_with_music
    )


@app.route("/insights")
def insights():
    db = get_db()

    emotion_stats = db.execute("""
        SELECT
            emotion,
            COUNT(*) AS total,
            ROUND(AVG(confidence), 2) AS average_confidence
        FROM analyses
        GROUP BY emotion
        ORDER BY total DESC, emotion ASC
    """).fetchall()

    total_analyses = db.execute("""
        SELECT COUNT(*) AS total
        FROM analyses
    """).fetchone()["total"]

    most_common_emotion = None

    if emotion_stats:
        most_common_emotion = emotion_stats[0]["emotion"]

    return render_template(
        "insights.html",
        emotion_stats=emotion_stats,
        total_analyses=total_analyses,
        most_common_emotion=most_common_emotion
    )


@app.route("/favorites", methods=["GET"])
def favorites():
    db = get_db()

    saved_songs = db.execute("""
        SELECT id, title, artist, emotion, created_at
        FROM favorites
        ORDER BY created_at DESC
    """).fetchall()

    songs_with_previews = []

    for saved_song in saved_songs:
        song = dict(saved_song)

        preview = get_itunes_preview(song["title"], song["artist"])

        song["preview_url"] = preview["preview_url"]
        song["itunes_url"] = preview["itunes_url"]
        song["spotify_url"] = (
            "https://open.spotify.com/search/"
            + quote_plus(f"{song['title']} {song['artist']}")
        )

        songs_with_previews.append(song)

    return render_template(
        "favorites.html",
        saved_songs=songs_with_previews
    )


@app.route("/favorites", methods=["POST"])
def add_favorite():
    title = request.form.get("title")
    artist = request.form.get("artist")
    emotion = request.form.get("emotion")

    if title and artist and emotion:
        db = get_db()

        db.execute(
            "INSERT INTO favorites (title, artist, emotion) VALUES (?, ?, ?)",
            (title, artist, emotion)
        )
        db.commit()

    return redirect(url_for("favorites"))


@app.route("/favorites/<int:favorite_id>/delete", methods=["POST"])
def delete_favorite(favorite_id):
    db = get_db()

    db.execute(
        "DELETE FROM favorites WHERE id = ?",
        (favorite_id,)
    )
    db.commit()

    return redirect(url_for("favorites"))


@app.route("/feedback", methods=["POST"])
def save_feedback():
    data = request.get_json(silent=True)

    if not data:
        return jsonify(error="Invalid feedback request."), 400

    title = data.get("title")
    artist = data.get("artist")
    feedback = data.get("feedback")

    if not title or not artist or feedback not in ["like", "dislike"]:
        return jsonify(error="Invalid feedback data."), 400

    db = get_db()

    db.execute(
        "INSERT INTO song_feedback (title, artist, feedback) VALUES (?, ?, ?)",
        (title, artist, feedback)
    )
    db.commit()

    return jsonify(message="Your feedback was saved.")


if __name__ == "__main__":
    initialize_database()
    app.run(debug=True)