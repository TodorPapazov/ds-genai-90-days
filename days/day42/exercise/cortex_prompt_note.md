# The same prompt in Cortex COMPLETE (Day 37 reference, do not run today)

Cortex COMPLETE takes the same messages array (system, user, assistant), so the v2 prompt you tune in Python moves to SQL unchanged.
Escape single quotes in the system prompt when you paste it into SQL, and keep the few-shot pairs as extra user/assistant objects.

```sql
SELECT NOTE_ID,
       SNOWFLAKE.CORTEX.COMPLETE('llama3.1-8b',
         [ {'role': 'system', 'content': '<the v2 SYSTEM text from prompts.py>'},
           {'role': 'user', 'content': 'Extract the fields from this note.\n<note>\n' ||
               REPLACE(REPLACE(NOTE_TEXT, '<note>', '[tag removed]'), '</note>', '[tag removed]') || '\n</note>'} ],
         {'temperature': 0, 'max_tokens': 300}) AS R
FROM DISCHARGE_NOTES WHERE NOTE_ID <= 5;
```

The test set idea is the same in both places: same notes, same model, one prompt change at a time, grade per field.
