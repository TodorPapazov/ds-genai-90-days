# DS + Gen AI · 90 Days (lesson site)
Static site (plain HTML/CSS/JS, no build step). It holds the 12-week roadmap, the day cards, search, a progress tracker (saved in browser localStorage), and per-day lessons with code, printables, PDF, podcast and social post.

## Run locally
`python -m http.server 8000` → http://localhost:8000 (fetch() needs a server; opening the file with file:// will not work).

## Publish with GitHub Pages
1. Put these files in your repo (`ds-genai-90-days`), either at the root or in `/docs`.
2. `git add . && git commit -m "Add lesson site" && git push`
3. GitHub → repo **Settings → Pages** → Source: *Deploy from a branch* → Branch `main`, folder `/ (root)` or `/docs` → Save.
4. Site goes live at `https://<username>.github.io/ds-genai-90-days/` within a minute or two. `.nojekyll` is included.

To add days, see ADD_DAY.md.
