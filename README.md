# TopicRadar

TopicRadar ist ein leichtgewichtiges Dashboard, das inhaltliche Überschneidungen im Team sichtbar macht. Die einzige Pflicht für Teammitglieder: **ca. alle 14 Tage ein kurzes Update** zu den aktuellen Themen eintragen.

## Was das Dashboard zeigt

- Welche Kolleg:innen thematisch am stärksten zu deinem aktuellen Fokus passen
- Gemeinsame Schlagwörter (Keywords) als Gesprächsaufhänger
- Wer seit mehr als 14 Tagen kein Update eingetragen hat (Erinnerung)

## Lokaler Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Dann im Browser `http://localhost:8501` öffnen.

## Datenmodell

Ein Update besteht aus:

- Name
- Datum
- Freitext (woran die Person arbeitet)

Alle Einträge werden in `data/updates.json` abgelegt.

## Berechnung der inhaltlichen Überschneidung

- Aus allen Texten werden TF-IDF-Vektoren erzeugt.
- Daraus wird die **Cosine Similarity** zwischen Kolleg:innen berechnet.
- Die Top-Matches werden im Dashboard angezeigt.

Das ist eine pragmatische, lokal laufende "Vector-DB-light"-Variante ohne externe Infrastruktur.
