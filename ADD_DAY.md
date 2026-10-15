# Adding a new day
1. Create `days/dayNN/` (two-digit, e.g. `day02`).
2. Drop in the 4 printable PNGs, the PDF and (optional) `dayNN_podcast.mp3`.
3. Copy `days/day01/lesson.json` to `days/dayNN/lesson.json` and edit: `title`, `concept` (HTML ok), `stats` (optional), `steps`, `snippets` (`lang`: python | sql | bash), `images` ([file, caption]), `pdf`, `audio`, `podcast` (paragraphs), `post`.
4. Add an entry to `days/days.json` → `days`: `{"day":2,"folder":"day02","title":"...","tags":["..."]}`.
5. Commit and push. Unlisted days show greyed out as "Coming soon". The audio player only appears when the mp3 exists.
