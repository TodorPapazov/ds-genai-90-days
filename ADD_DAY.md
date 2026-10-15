# Adding a new day
1. Create `days/dayNN/` (two-digit, e.g. `day02`).
2. Drop in the 4 printable PNGs, the PDF, the podcast `dayNN_podcast.mp3` (edge-tts, voice `en-US-AndrewNeural`: `python3 -m edge_tts --voice en-US-AndrewNeural --rate=-5% -f script.txt --write-media dayNN_podcast.mp3`), `lesson.md` (full lesson in Markdown) and `dayNN_all_materials.zip` (`zip dayNN_all_materials.zip *.png *.pdf *.mp3 lesson.md`).
3. Copy `days/day01/lesson.json` to `days/dayNN/lesson.json` and edit: `title`, `concept` (HTML ok), `stats` (optional), `steps`, `snippets` (`lang`: python | sql | bash), `images` ([file, caption]), `pdf`, `audio` (mp3 filename), `podcast` (paragraphs = full script), `post`, `zip` (zip filename), `md` (`lesson.md`). These feed the player + MP3 download and the Downloads section.
4. Add an entry to `days/days.json` → `days`: `{"day":2,"folder":"day02","title":"...","tags":["..."]}`.
5. Commit and push. Unlisted days show greyed out as "Coming soon". The audio player only appears when the mp3 exists.
