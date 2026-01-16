"""Test Official mode on deployed HuggingFace Space"""
import requests
import json

SPACE_URL = "https://wbw2000-alpamayo-r1-demo.hf.space"

def test_api_available():
    """Check if Space API is available"""
    try:
        resp = requests.get(f"{SPACE_URL}/config", timeout=30)
        if resp.status_code == 200:
            print("[OK] Space API is available")
            return True
        else:
            print(f"[FAIL] Space API returned status {resp.status_code}")
            return False
    except Exception as e:
        print(f"[FAIL] Cannot reach Space: {e}")
        return False

def test_mode_change():
    """Test switching to Official mode"""
    try:
        # Gradio API call for mode change
        payload = {
            "data": ["Official (Paper-Comparable)"],
            "fn_index": 1,  # on_mode_change function
            "session_hash": "test123"
        }
        resp = requests.post(
            f"{SPACE_URL}/gradio_api/call/on_mode_change",
            json={"data": ["Official (Paper-Comparable)"]},
            timeout=30
        )
        print(f"Mode change response: {resp.status_code}")
        if resp.status_code == 200:
            print("[OK] Mode change endpoint works")
            return True
    except Exception as e:
        print(f"Mode change test: {e}")
    return False

def get_space_status():
    """Get current Space status"""
    try:
        resp = requests.get(f"{SPACE_URL}/", timeout=30)
        if resp.status_code == 200:
            print(f"[OK] Space is running (HTTP 200)")
            return True
    except Exception as e:
        print(f"Status check: {e}")
    return False

if __name__ == "__main__":
    print("=" * 50)
    print("Testing Alpamayo-R1 Official Mode")
    print("=" * 50)

    print("\n1. Checking Space availability...")
    get_space_status()

    print("\n2. Checking API configuration...")
    test_api_available()

    print("\n3. Testing mode change endpoint...")
    test_mode_change()

    print("\n" + "=" * 50)
    print("Manual Testing Instructions:")
    print("=" * 50)
    print("""
To test Official mode in browser:

1. Visit: https://huggingface.co/spaces/wbw2000/alpamayo-r1-demo

2. Click "Load Model" and wait for model to load (~3 min)

3. Select "Official (Paper-Comparable)" mode

4. Enter a Clip ID, e.g.:
   030c760c-ae38-49aa-9ad8-f5650a545d26

5. Set t0 (microseconds): 5100000

6. Click "Run Inference"

Expected Results:
- If HF_TOKEN is configured: minADE6@6.4s metric displayed
- If HF_TOKEN missing: Error message about token requirement
""")
