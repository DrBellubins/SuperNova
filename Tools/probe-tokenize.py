#!/usr/bin/env python3
"""Probe the reference server's /tokenize request and response shape (one model load)."""
import json, os, signal, subprocess, sys, time, urllib.request

MODEL = "/home/justin/Documents/AI/Models/SenseLoop/SenseNova-Loop-9B-Q4_K_M.gguf"
SERVER = "/home/justin/Documents/Github/llama.cpp-opensensenova/build/bin/llama-server"
PORT = 8124


def post(path, obj, timeout=120):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}",
                                data=json.dumps(obj).encode(),
                                headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def wait_ready(deadline):
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(1.0)
    return False


log = open("/tmp/probe-server.log", "w")
p = subprocess.Popen([SERVER, "-m", MODEL, "-c", "2048", "-ngl", "99", "--port", str(PORT), "-t", "4"],
                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
try:
    if not wait_ready(time.time() + 240):
        sys.exit("not healthy, see /tmp/probe-server.log")
    for body in ({"prompt": "The capital of France is"},
                 {"content": "The capital of France is"},
                 {"prompt": "The capital of France is", "add_special": False}):
        try:
            r = post("/tokenize", body)
            print(f"REQ {json.dumps(body)}\n  -> {json.dumps(r)[:300]}")
        except Exception as e:
            print(f"REQ {json.dumps(body)}\n  -> ERROR {e}")
    print("/props keys:", sorted(post("/props", {}))[:20])
finally:
    os.killpg(os.getpgid(p.pid), signal.SIGTERM)
    try:
        p.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(p.pid), signal.SIGKILL)
    log.close()
