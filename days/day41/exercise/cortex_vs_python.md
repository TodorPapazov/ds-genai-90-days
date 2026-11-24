# Same job, two places: Cortex COMPLETE in SQL (Day 37) vs an LLM API from Python (Day 41)

Reference only. Do not run the SQL today; it is the Day 37 pattern, shown so you can map each part.

```sql
-- Day 37 (Snowflake): the model runs next to the table. One statement, many rows.
SELECT NOTE_ID,
       SNOWFLAKE.CORTEX.COMPLETE('llama3.1-8b',
         [ {'role': 'system', 'content': 'You extract fields from simulated discharge notes.'},
           {'role': 'user',   'content': 'Extract the fields: ' || NOTE_TEXT} ],
         {'temperature': 0, 'max_tokens': 120}) AS R      -- returns choices + usage, like an API
FROM DISCHARGE_NOTES WHERE NOTE_ID <= 5;
```

```python
# Day 41 (Python): you send each request yourself, over HTTPS, with your own key from an env var.
resp = lc.chat_with_retries(client, messages_for(note), model=lc.MODEL,
                            temperature=0, max_tokens=300, response_format=RESPONSE_FORMAT)
fields = json.loads(resp["choices"][0]["message"]["content"])
```

| | Cortex COMPLETE / AI_COMPLETE (SQL) | LLM API from Python |
|---|---|---|
| Where the note goes | Stays inside your Snowflake account and its governance | Leaves your machine for the provider's servers |
| Who can call it | Snowflake roles (CORTEX_USER) | Anyone holding the API key: guard it |
| Loop over rows | The SQL engine does it: one SELECT over a table | You write the loop, batching and rate-limit handling |
| Retries, timeouts | Handled for you inside the query | Your code: backoff on 429 / 5xx, a timeout on every call |
| Same knobs | messages with roles, temperature, max_tokens, JSON schema | messages with roles, temperature, max_tokens, JSON schema |
| Billing | Snowflake credits per token (plus the warehouse) | Provider dollars per token (input and output priced separately) |
| Models | The Cortex model list in your region | Whatever the provider offers; switch with base URL + model name |
| Results land in | A table (CTAS) | Files (JSONL, CSV), then pandas or a load back into the warehouse |
| Good fit | Batch enrichment of warehouse tables, governed data | Apps, notebooks, pipelines outside the warehouse, provider choice |

Rule for real work: patient data goes only where your employer has approved it (for an outside API that usually means a signed BAA and an approved account). This exercise uses simulated notes only.
