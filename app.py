from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = Path("data/updates.json")


def load_updates() -> pd.DataFrame:
    if not DATA_PATH.exists():
        return pd.DataFrame(columns=["name", "date", "text"])

    with DATA_PATH.open("r", encoding="utf-8") as f:
        records = json.load(f)

    df = pd.DataFrame(records)
    if df.empty:
        return pd.DataFrame(columns=["name", "date", "text"])

    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def save_updates(df: pd.DataFrame) -> None:
    payload = df.copy()
    payload["date"] = payload["date"].astype(str)
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with DATA_PATH.open("w", encoding="utf-8") as f:
        json.dump(payload.to_dict(orient="records"), f, indent=2, ensure_ascii=False)


def latest_update_per_person(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    latest_idx = df.groupby("name")["date"].idxmax()
    latest = df.loc[latest_idx].copy()
    latest = latest.sort_values("name").reset_index(drop=True)
    return latest


def build_similarity(df_latest: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], TfidfVectorizer]:
    names = df_latest["name"].tolist()
    texts = df_latest["text"].tolist()

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
    matrix = vectorizer.fit_transform(texts)
    sim = cosine_similarity(matrix)

    sim_df = pd.DataFrame(sim, index=names, columns=names)
    return sim_df, names, vectorizer


def top_keywords(vectorizer: TfidfVectorizer, text: str, top_n: int = 6) -> List[str]:
    row = vectorizer.transform([text]).toarray()[0]
    if row.sum() == 0:
        return []

    features = vectorizer.get_feature_names_out()
    ranked_idx = row.argsort()[::-1]
    return [features[i] for i in ranked_idx[:top_n] if row[i] > 0]


def pairs_with_overlap(sim_df: pd.DataFrame, threshold: float = 0.1) -> pd.DataFrame:
    rows = []
    people = list(sim_df.index)
    for i, left in enumerate(people):
        for right in people[i + 1 :]:
            score = float(sim_df.loc[left, right])
            if score >= threshold:
                rows.append({"Person A": left, "Person B": right, "Ähnlichkeit": round(score, 3)})

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    return out.sort_values("Ähnlichkeit", ascending=False).reset_index(drop=True)


def days_since(d: date) -> int:
    return (date.today() - d).days


def app() -> None:
    st.set_page_config(page_title="TopicRadar", page_icon="🧭", layout="wide")
    st.title("🧭 TopicRadar")
    st.caption("Finde inhaltliche Überschneidungen im Team – mit einem kurzen 14-Tage-Update pro Person.")

    df_all = load_updates()

    with st.sidebar:
        st.header("Neues Update")
        with st.form("new-update"):
            name = st.text_input("Name")
            update_date = st.date_input("Datum", value=date.today())
            text = st.text_area("Woran arbeitest du aktuell?", height=140)
            submitted = st.form_submit_button("Speichern")

        if submitted:
            if not name.strip() or not text.strip():
                st.error("Bitte Name und Beschreibung ausfüllen.")
            else:
                next_row = pd.DataFrame(
                    [{"name": name.strip(), "date": update_date, "text": text.strip()}]
                )
                df_all = pd.concat([df_all, next_row], ignore_index=True)
                save_updates(df_all)
                st.success("Update gespeichert.")
                st.rerun()

    if df_all.empty:
        st.info("Noch keine Daten vorhanden. Trage links das erste Update ein.")
        return

    latest_df = latest_update_per_person(df_all)

    c1, c2, c3 = st.columns(3)
    c1.metric("Teammitglieder", len(latest_df))
    c2.metric("Gesamte Updates", len(df_all))
    stale_count = int((latest_df["date"].apply(days_since) > 14).sum())
    c3.metric("Älter als 14 Tage", stale_count)

    st.subheader("14-Tage-Status")
    status = latest_df[["name", "date"]].copy()
    status["Tage seit letztem Update"] = status["date"].apply(days_since)
    status["Status"] = status["Tage seit letztem Update"].apply(
        lambda d: "⚠️ Update fällig" if d > 14 else "✅ Aktuell"
    )
    st.dataframe(status.sort_values("Tage seit letztem Update", ascending=False), use_container_width=True)

    if len(latest_df) < 2:
        st.warning("Mindestens zwei Personen mit Updates sind nötig, um Überschneidungen zu berechnen.")
        return

    sim_df, names, vectorizer = build_similarity(latest_df)

    st.subheader("Thematische Überschneidungen")
    st.dataframe(sim_df.style.format("{:.2f}"), use_container_width=True)

    overlap_pairs = pairs_with_overlap(sim_df, threshold=0.08)
    if overlap_pairs.empty:
        st.info("Aktuell keine signifikanten Überschneidungen über dem Schwellwert gefunden.")
    else:
        st.dataframe(overlap_pairs, use_container_width=True)

    st.subheader("Dein persönlicher Match")
    selected_person = st.selectbox("Person auswählen", names)
    ranked = (
        sim_df.loc[selected_person]
        .drop(index=selected_person)
        .sort_values(ascending=False)
        .reset_index()
    )
    ranked.columns = ["Kolleg:in", "Ähnlichkeit"]

    if not ranked.empty:
        top = ranked.iloc[0]
        st.success(
            f"Bester Match für **{selected_person}**: **{top['Kolleg:in']}** "
            f"(Score: {top['Ähnlichkeit']:.2f})"
        )

    st.dataframe(ranked.style.format({"Ähnlichkeit": "{:.2f}"}), use_container_width=True)

    selected_text = latest_df.loc[latest_df["name"] == selected_person, "text"].iloc[0]
    keywords = top_keywords(vectorizer, selected_text)
    st.markdown(
        "**Top-Keywords aus dem letzten Update:** "
        + (", ".join(keywords) if keywords else "Keine Keywords erkannt")
    )

    with st.expander("Rohdaten anzeigen"):
        st.dataframe(df_all.sort_values("date", ascending=False), use_container_width=True)


if __name__ == "__main__":
    app()
