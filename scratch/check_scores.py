import httpx

BASE = "http://localhost:7860"
TASKS = ["single_quarter_survival", "four_quarter_growth", "adversarial_resilience"]

print("\n--- Current Grader Scores (Post-Reset) ---")
for task_id in TASKS:
    try:
        r = httpx.post(f"{BASE}/reset", params={"task_id": task_id})
        sid = r.json()["session_id"]
        rg = httpx.get(f"{BASE}/grade", params={"session_id": sid})
        score = rg.json()["score"]
        print(f"{task_id:25}: {score:.4f}")
    except Exception as e:
        print(f"{task_id:25}: Error - {e}")
print("------------------------------------------\n")
