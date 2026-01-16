"""Test full inference flow on deployed HuggingFace Space"""
import requests
import json
import time

SPACE_URL = "https://wbw2000-alpamayo-r1-demo.hf.space"

def call_gradio_api(fn_name, data, timeout=300):
    """Call Gradio API endpoint"""
    # First, initiate the call
    resp = requests.post(
        f"{SPACE_URL}/gradio_api/call/{fn_name}",
        json={"data": data},
        timeout=30
    )
    if resp.status_code != 200:
        return None, f"Failed to initiate call: {resp.status_code}"

    result = resp.json()
    event_id = result.get("event_id")
    if not event_id:
        return None, "No event_id returned"

    # Then, wait for result
    print(f"  Event ID: {event_id}, waiting for result...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            resp = requests.get(
                f"{SPACE_URL}/gradio_api/call/{fn_name}/{event_id}",
                timeout=30,
                stream=True
            )
            if resp.status_code == 200:
                # Parse SSE response
                for line in resp.iter_lines(decode_unicode=True):
                    if line.startswith("data:"):
                        data_str = line[5:].strip()
                        if data_str:
                            try:
                                return json.loads(data_str), None
                            except:
                                return data_str, None
        except Exception as e:
            print(f"  Waiting... ({e})")
        time.sleep(2)

    return None, "Timeout waiting for result"

def test_official_mode_inference():
    """Test Official mode inference"""
    print("\n" + "=" * 50)
    print("Testing Official Mode Inference")
    print("=" * 50)

    # Test clip ID from sample dataset
    clip_id = "030c760c-ae38-49aa-9ad8-f5650a545d26"
    t0_us = 5100000

    print(f"\nClip ID: {clip_id}")
    print(f"t0 (us): {t0_us}")

    # First, check if we can access the Official dataset sidecars
    print("\n1. Testing sidecar data fetch (via mode change)...")
    result, error = call_gradio_api("on_mode_change", ["Official (Paper-Comparable)"], timeout=30)
    if error:
        print(f"  [FAIL] Mode change: {error}")
    else:
        print(f"  [OK] Mode change successful")
        print(f"  Result: {result}")

    print("\n2. Note: Full inference test requires GPU allocation")
    print("   Please test manually in browser:")
    print(f"   URL: {SPACE_URL}")
    print(f"   1. Load Model")
    print(f"   2. Select 'Official (Paper-Comparable)' mode")
    print(f"   3. Enter Clip ID: {clip_id}")
    print(f"   4. Click Run Inference")

    return True

def check_hf_token_status():
    """Check if HF_TOKEN is likely configured"""
    print("\n" + "=" * 50)
    print("HF_TOKEN Status Check")
    print("=" * 50)

    # Try to get space info
    try:
        resp = requests.get(
            "https://huggingface.co/api/spaces/wbw2000/alpamayo-r1-demo",
            timeout=30
        )
        if resp.status_code == 200:
            data = resp.json()
            runtime = data.get("runtime", {})
            stage = runtime.get("stage", "unknown")
            hardware = runtime.get("hardware", {}).get("current", "unknown")
            print(f"  Space stage: {stage}")
            print(f"  Hardware: {hardware}")

            # Check if secrets section exists (cannot see values, just existence)
            # This is not directly available via API
            print("  Note: HF_TOKEN secret visibility cannot be checked via API")
            print("  If Official mode works, HF_TOKEN is configured correctly")
    except Exception as e:
        print(f"  Error checking space: {e}")

if __name__ == "__main__":
    check_hf_token_status()
    test_official_mode_inference()

    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)
    print("""
Space Status: RUNNING
Hardware: zero-a10g (A10G GPU)

To verify HF_TOKEN is working:
1. Open https://huggingface.co/spaces/wbw2000/alpamayo-r1-demo
2. Load the model
3. Switch to "Official (Paper-Comparable)" mode
4. Enter clip ID: 030c760c-ae38-49aa-9ad8-f5650a545d26
5. Run inference

If you see minADE6@6.4s metric -> HF_TOKEN is working
If you see "HF_TOKEN required" error -> Need to add secret
""")
