#!/usr/bin/env python3
"""Offline quantitative benchmark: no tmux, simulator, or ROS nodes."""
import argparse, csv, json, math, subprocess
from pathlib import Path
import yaml
from quantitative_test import FIELDS, ROOT, finite_mean, points_length, report

def run_pair(map_id, shape, start, goal, free_start_yaw=False, start_yaw_deg=0.0):
    command = ["rosrun", "mighty", "benchmark_risk_local_backend", "--offline-request",
               map_id, shape, *map(str, start), *map(str, goal)]
    if start_yaw_deg != 0.0:
        command.extend(["--start-yaw-deg",str(start_yaw_deg)])
        free_start_yaw=False
    elif free_start_yaw: command.append("--free-start-yaw")
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    try: data = json.loads(completed.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        return {"start":start,"goal":goal,"status":"FAILED","reason":completed.stderr or completed.stdout,
                "frontend":[],"middle":[],"backend":[]}
    middle = [p[:4] for p in data["middle"]]; backend = [p[:4] for p in data["backend"]]
    return {"start":start,"goal":goal,"status":"OK" if data["success"] and completed.returncode == 0 else "FAILED","reason":data["reason"],
            "start_yaw":data.get("start_yaw",0.0),"start_yaw_deg":start_yaw_deg,"free_start_yaw":free_start_yaw,
            "optimization_status":data.get("optimization_status"),
            "refinement_timed_out":data.get("refinement_timed_out"),
            "frontend":data["frontend"],"middle":middle,"backend":backend,
            "frontend_length":data["frontend_length"],"frontend_ms":data["frontend_ms"],
            "middle_length":points_length(middle),"middle_ms":data["middle_ms"],
            "backend_length":data["backend_length"],"backend_ms":data["backend_ms"],
            "mean_speed":data["backend_length"]/data["duration_s"] if data["duration_s"] else None}

def main():
    p=argparse.ArgumentParser(); p.add_argument("--results",type=Path,default=ROOT/"scripts/results"); p.add_argument("--case"); p.add_argument("--pair-limit",type=int)
    p.add_argument("--free-start-yaw",action="store_true",help="Choose a safe yaw where no nonzero fixture yaw is specified; default uses fixture yaw or zero.")
    args=p.parse_args()
    cases=yaml.safe_load((ROOT/"scripts/start2end.yaml").read_text())["cases"]
    if args.case: cases={args.case:cases[args.case]}
    for case in cases.values():
        yaws=case.setdefault("start_yaw_deg",[0.0]*len(case["pairs"]))
        if len(yaws) != len(case["pairs"]) or not all(math.isfinite(float(yaw)) for yaw in yaws):
            p.error("start_yaw_deg must contain one finite angle per pair")
    if args.pair_limit: cases={k:{**v,"pairs":v["pairs"][:args.pair_limit],"start_yaw_deg":v["start_yaw_deg"][:args.pair_limit]} for k,v in cases.items()}
    args.results.mkdir(parents=True,exist_ok=True)
    ids=[int(path.name[3:]) for path in args.results.glob("val*") if path.name[3:].isdigit()]
    run_id=max(ids,default=0)+1; folder=args.results/f"val{run_id}"; folder.mkdir()
    all_rows=[]
    results={}
    for caseid,case in cases.items():
        print(f"[val{run_id}] {caseid}",flush=True)
        rows=[run_pair(case["map"],case["shape"],start,goal,args.free_start_yaw,yaw)
              for (start,goal),yaw in zip(case["pairs"],case["start_yaw_deg"])]
        results[caseid]=rows
        (folder/"results.json").write_text(json.dumps(results,ensure_ascii=False,indent=2))
        map_folder=folder/case["map"]; map_folder.mkdir(exist_ok=True)
        report(map_folder/f"{caseid}.html",case,rows); all_rows.extend(rows)
    summary=args.results/"summary.csv"; new_file=not summary.exists()
    with summary.open("a",newline="",encoding="utf-8-sig") as f:
        writer=csv.DictWriter(f,fieldnames=FIELDS)
        if new_file: writer.writeheader()
        writer.writerow({FIELDS[0]:f"val{run_id}",FIELDS[1]:finite_mean(all_rows,"frontend_length"),FIELDS[2]:finite_mean(all_rows,"frontend_ms"),FIELDS[3]:finite_mean(all_rows,"middle_length"),FIELDS[4]:finite_mean(all_rows,"middle_ms"),FIELDS[5]:finite_mean(all_rows,"backend_length"),FIELDS[6]:finite_mean(all_rows,"backend_ms"),FIELDS[7]:finite_mean(all_rows,"mean_speed")})
    successful=sum(row["status"] == "OK" for row in all_rows)
    print(f"[val{run_id}] success={successful}/{len(all_rows)}; results={folder}",flush=True)
    return 0 if all_rows and successful == len(all_rows) else 1
if __name__=="__main__": raise SystemExit(main())
