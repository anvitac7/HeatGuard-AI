"""
HeatGuard AI load test (Exp 7 / Exp 8).
Usage:  python loadtest.py --label baseline --duration 60 --concurrency 5
Appends one row per run to results.csv. All numbers are measured, not simulated.
Requires: pip install requests
"""
import argparse, csv, os, random, statistics, threading, time
from collections import Counter
import requests

CITIES = ["Delhi", "Mumbai", "Chennai", "Kolkata", "Bengaluru", "Pune", "Ahmedabad"]

def build_requests(base):
    """Each item: (name, method, url, json_body, expected_status_group)"""
    ok = [
        ("predict-all", "POST", f"{base}/api/predict-all", {"date": "2024-05-28"}, 2),   # POST confirmed via 405 on GET
        ("healthz", "GET", f"{base}/healthz", None, 2),
    ]
    for c in CITIES[:3]:
        ok.append(("advisory", "POST", f"{base}/api/advisory",
                   {"city": c, "date": "2024-05-28", "audience": "Citizen"}, 2))
    bad = [  # deliberate invalid inputs: a 4xx here counts as correct handling
        ("invalid-city", "POST", f"{base}/api/advisory",
         {"city": "Atlantis", "date": "2024-05-28", "audience": "Citizen"}, 4),
        ("invalid-date", "POST", f"{base}/api/advisory",
         {"city": "Delhi", "date": "2030-01-01", "audience": "Citizen"}, 4),
    ]
    return ok, bad

def worker(stop_at, ok, bad, results, lock):
    s = requests.Session()
    while time.time() < stop_at:
        name, method, url, body, expect = random.choice(bad) if random.random() < 0.1 else random.choice(ok)
        t0 = time.perf_counter()
        try:
            r = s.request(method, url, json=body, timeout=60)
            status = r.status_code
        except Exception:
            status = 0
        dt = (time.perf_counter() - t0) * 1000
        correct = (status // 100) == expect
        with lock:
            results.append((name, dt, status, correct))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8081")
    ap.add_argument("--label", required=True)
    ap.add_argument("--duration", type=int, default=60)
    ap.add_argument("--concurrency", type=int, default=5)
    a = ap.parse_args()

    ok, bad = build_requests(a.base)
    results, lock = [], threading.Lock()
    start = time.time()
    stop_at = start + a.duration
    threads = [threading.Thread(target=worker, args=(stop_at, ok, bad, results, lock))
               for _ in range(a.concurrency)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    elapsed = time.time() - start

    lat = sorted(r[1] for r in results)
    n = len(lat)
    p = lambda q: lat[min(n - 1, int(q * n))]
    errors = sum(1 for r in results if not r[3])
    row = {
        "label": a.label, "concurrency": a.concurrency, "duration_s": round(elapsed, 1),
        "requests": n, "rps": round(n / elapsed, 2),
        "mean_ms": round(statistics.mean(lat), 1), "p50_ms": round(p(0.50), 1),
        "p95_ms": round(p(0.95), 1), "max_ms": round(lat[-1], 1),
        "unexpected_responses": errors,
    }
    print(row)
    print("Breakdown (endpoint, status): count")
    for (name, status), c in sorted(Counter((r[0], r[2]) for r in results).items()):
        print(f"  {name:14s} {status}: {c}")
    new = not os.path.exists("results.csv")
    with open("results.csv", "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if new:
            w.writeheader()
        w.writerow(row)

if __name__ == "__main__":
    main()