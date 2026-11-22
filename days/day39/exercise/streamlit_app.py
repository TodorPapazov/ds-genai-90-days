# Day 39: a tiny Streamlit in Snowflake chat over the READMIT_ANALYST semantic view (SIMULATED data).
# Snowsight: Projects > Streamlit > + Streamlit App, warehouse DS90_XS, location DS90.DAY39, paste this file, Run.
# The app runs as YOUR role inside Snowflake, so there is no password or token anywhere.
# Cost: every question = 1 Cortex Analyst message, plus warehouse time while the app is open. Close the tab when done.
import json
import _snowflake                                  # available inside Streamlit in Snowflake only
import streamlit as st
from snowflake.snowpark.context import get_active_session

VIEW = "DS90.DAY39.READMIT_ANALYST"
MAX_MESSAGES = 15                                  # guardrail: per-session message cap
BLOCKED = ("INSERT", "UPDATE", "DELETE", "MERGE", "CREATE", "ALTER", "DROP", "TRUNCATE", "GRANT", "CALL")
session = get_active_session()
st.title("Readmissions Q&A (simulated data)")
st.caption("Cortex Analyst + semantic view. Practice data only, not real patients.")
st.session_state.setdefault("n", 0)

question = st.chat_input("Ask about readmissions, payers, DRGs, counties ...")
if question:
    if st.session_state.n >= MAX_MESSAGES:
        st.warning("Message cap reached for this session.")
        st.stop()
    body = {"messages": [{"role": "user", "content": [{"type": "text", "text": question}]}], "semantic_view": VIEW}
    resp = _snowflake.send_snow_api_request("POST", "/api/v2/cortex/analyst/message", {}, {}, body, None, 50000)
    st.session_state.n += 1
    content = json.loads(resp["content"])
    if resp["status"] >= 400:
        st.error(f"Analyst error {resp['status']}: {content.get('message')}")
        st.stop()
    st.chat_message("user").write(question)
    with st.chat_message("assistant"):
        for c in content["message"]["content"]:
            if c["type"] == "text":
                st.write(c["text"])
            elif c["type"] in ("suggestion", "suggestions"):
                for s in c.get("suggestions", []):
                    st.write("- " + s)
            elif c["type"] == "sql":
                sql = c["statement"]
                st.code(sql, language="sql")
                if not sql.lstrip().upper().startswith(("SELECT", "WITH")) or any(f" {k} " in f" {sql.upper()} " for k in BLOCKED):
                    st.error("Blocked: only read-only SELECT statements run here.")
                else:
                    df = session.sql(f"SELECT * FROM (\n{sql.strip().rstrip(';')}\n) LIMIT 1000").to_pandas()
                    st.dataframe(df)
                    if 1 < len(df) <= 40 and df.shape[1] >= 2:
                        st.bar_chart(df.set_index(df.columns[0]).select_dtypes("number"))
    st.caption(f"Messages this session: {st.session_state.n} / {MAX_MESSAGES}")
