# Day 40: guided Cortex Code (CoCo) session, about 25 minutes

Snowflake's AI coding agent was launched as **Cortex Code**. The docs now call it **Snowflake CoCo**; the usage views and parameters still say `CORTEX_CODE`. It comes in several forms: CoCo in Snowsight (a panel in the web UI, strongest inside Workspaces), the CoCo CLI (`cortex` in a terminal), CoCo Desktop, and previews for Excel and Google Sheets. Docs checked Nov 23, 2026.

Honest limits of this guide: the lesson author could not open your Snowsight. The clicks below are the ones the Snowflake docs describe. If your screen looks different, trust the screen and the docs, and write down what you saw.

## Before you start (5 minutes)

- [ ] `01_setup.sql` parts B and C: your role has `SNOWFLAKE.CORTEX_USER` (through PUBLIC by default), cross-region inference is not `DISABLED`, and you set a small `CORTEX_CODE_SNOWSIGHT_DAILY_EST_CREDIT_LIMIT_PER_USER` on yourself.
- [ ] Trial note from the docs: on self-service trial accounts, AI features stay off until an account admin adds a credit card. That does not end the trial. If Days 37 to 39 worked, this is already done.
- [ ] Open the CoCo panel: the docs say to select the CoCo icon in the lower-right corner of Snowsight; the panel opens on the right.
- [ ] Open a workspace (the docs path is Projects, then Workspaces) and add the Day 40 SQL files, so CoCo can see the file you are looking at.
- [ ] Look at the model selector (docs: lower right of the message box). Auto options pick a model for you; Auto Efficient adds a small routing fee. Write down which one you used.

Every prompt is billed by tokens (input plus output, at the chosen model's rate). Short, specific prompts with fully qualified names (`DS90.DAY40.SHORTLIST`) are cheaper and more accurate. Type `@` to attach a table or schema as context.

## The prompts

Log each one in `DS90.DAY40.COCO_LOG` (insert template at the bottom) or in the README table.

**P1. Explore, in a read-only chat.** Docs: in the CoCo panel select +, then Restrict this chat, and choose read-only. Then ask:
> List the schemas DAY36 to DAY40 in database DS90 and the tables, models, semantic views and Cortex Search services in them. Which of these can use credits while I am not using them?

Check against `01_setup.sql` part A. Good answer: tables cost only storage; a semantic view and a registered model are metadata plus small storage; a warehouse bills only while running; a resumed Cortex Search service bills serving credits even with no queries.

**P2. Guardrail test, same restricted chat.**
> Create a table DS90.DAY40.RSS_TEST (x INT).

In a read-only chat this should be refused or fail. Harmless if it works (drop it), but then the restriction was not active: note it.

**P3. Explain.** Open `03_extract.sql`, highlight part D, and use the Explain quick action (docs: highlighting SQL offers Quick Edit, Format, Add to Chat, Explain). Then ask:
> How many AI_COMPLETE calls does this statement make, and how does its token use compare with running it on all 300 rows of DS90.DAY40.DISCHARGES?

Good answer: 60 calls, about one fifth of the tokens of 300 rows (lesson-box estimate: about 9,800 vs 48,000 input tokens). If it quotes a credit price, check it against the Service Consumption Table before you believe it.

**P4. Generate, then prove it.** In a new, normal chat:
> Using @DS90.DAY40.SHORTLIST and @DS90.DAY40.NOTE_FIELDS_PARSED, write a query that scores each patient as RISK_SCORE + 0.10 if LIVES_ALONE_AI + 0.12 if FOLLOWUP_BOOKED_AI is false, and returns the top 25 with ENCOUNTER_ID, SERVICE_LINE, and the score.

Run its query, then check it against the reference (expect zero rows both ways):
```sql
SELECT ENCOUNTER_ID FROM (<CoCo query>) MINUS SELECT ENCOUNTER_ID FROM DS90.DAY40.FOLLOWUP_WORKLIST;
SELECT ENCOUNTER_ID FROM DS90.DAY40.FOLLOWUP_WORKLIST MINUS SELECT ENCOUNTER_ID FROM (<CoCo query>);
```
Watch for: ties ordered differently, NULL booleans treated as false or true, `LIMIT 25` without a tie-breaker.

**P5. Fix an error.** Open `cortex_code_broken.sql` and run query 1. It fails (wrong column name, missing GROUP BY column). Use the Fix button in the results grid (docs). Read the diff before you accept it.

**P6. Review a query that runs but is wrong.** Highlight query 2, Add to Chat, and ask:
> Is this result correct for the share of the worklist with no follow-up booked?

The bug: it divides by all 300 discharges instead of the 25 on the list. Did CoCo catch it without a hint? This is the lesson of the day: Fix catches errors; only review catches wrong logic.

**P7. Edit with a diff.**
> Add a short comment block at the top of 05_worklist.sql saying the data is simulated and the weights are teaching choices, not clinical guidance.

Review the diff view, then accept or reject. Do not let it change the SQL itself.

**P8. Cost question.** The docs list a bundled `cost-intelligence` skill (type `/cost-intelligence`, or just ask in plain language):
> Which service types used the most credits in this account since November 16, 2026? Show the SQL you used.

Compare with `07_cost_check.sql` part A. Same view (`METERING_DAILY_HISTORY`)? Same total?

## After the session

- [ ] `07_cost_check.sql` part F: your CoCo requests, tokens and credits (`CORTEX_CODE_SNOWSIGHT_USAGE_HISTORY`, latency up to about 1 hour).
- [ ] Optional: remove your personal cap with `ALTER USER TODORPAPAZOV UNSET CORTEX_CODE_SNOWSIGHT_DAILY_EST_CREDIT_LIMIT_PER_USER;` or keep it.

## CoCo CLI: read, do not install today

The CLI (`cortex`) runs in a local terminal, can read and write local files, run git and SQL, and asks before each action in plan mode (`--plan`). It uses `~/.snowflake/connections.toml`, the same file as the Snowflake CLI, and the docs list the Snowflake CLI as a prerequisite. Windows native (PowerShell installer) and WSL are both supported.

The CLI docs say it is **not available on standard Snowflake trial accounts** (accounts made at trial.snowflake.com). It needs a paid account or a separate CoCo CLI trial, and that separate trial turns into a paid subscription after 30 days unless you cancel it. So skip it on this trial. If you use it later on a paid account: start with `cortex -c <connection> --plan`, keep secrets out of the repo, and never use `--bypass` (it approves every tool call).

## COCO_LOG insert template

```sql
INSERT INTO DS90.DAY40.COCO_LOG (PROMPT_ID, SURFACE, PROMPT, WHAT_IT_DID, RAN_SQL, MATCHED_REFERENCE, MY_EDITS)
VALUES ('P4', 'Snowsight', 'top 25 worklist query', 'wrote a QUALIFY query, ordered by score', TRUE, 'yes, 0 rows MINUS', 'none');
```
