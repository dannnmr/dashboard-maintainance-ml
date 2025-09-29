# etl/bronze/extract.py
import os
import time
import pandas as pd
import requests
from requests_ntlm import HttpNtlmAuth
from dotenv import load_dotenv
import urllib3
from datetime import timedelta

# Disable SSL warnings for self-signed certificates (only if your PI server needs it)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

load_dotenv()

# --- Environment configuration ---
PI_USERNAME = os.getenv("PI_USERNAME")
PI_PASSWORD = os.getenv("PI_PASSWORD")
PI_SERVER   = os.getenv("PI_SERVER")
BASE_URL    = f"https://{PI_SERVER}/piwebapi"
HEADERS     = {"Content-Type": "application/json"}
print(PI_USERNAME)

def obtener_webid(tag_name: str) -> str:
    """
    Return the WebID for a given PI tag.
    """
    url = f"{BASE_URL}/points?path=\\\\{PI_SERVER}\\{tag_name}"
    resp = requests.get(url, auth=HttpNtlmAuth(PI_USERNAME, PI_PASSWORD), verify=False)
    resp.raise_for_status()
    return resp.json()["WebId"]


def generar_rangos_fechas(start_date_str: str, end_date_str: str, delta_dias: int = 15):
    """
    Generate half-open time ranges [start, end] split in windows of 'delta_dias'.
    """
    start_date = pd.to_datetime(start_date_str)
    end_date   = pd.to_datetime(end_date_str)
    rangos = []
    current_start = start_date

    while current_start < end_date:
        current_end = min(current_start + timedelta(days=delta_dias), end_date)
        rangos.append((current_start.isoformat(), current_end.isoformat()))
        current_start = current_end + timedelta(seconds=1)

    return rangos


def obtener_datos_hist_pag(webid: str, start: str, end: str, max_per_request: int = 800_000) -> pd.DataFrame:
    """
    Pull recorded data from PI Web API in pages until no more items are returned.
    Returns a DataFrame with columns: ['timestamp', 'value'].
    """
    all_items = []
    current_start = start

    while True:
        url = f"{BASE_URL}/streams/{webid}/recorded"
        params = {
            "startTime": current_start,
            "endTime": end,
            "maxCount": max_per_request,
            "boundaryType": "Inside",
        }
        resp = requests.get(
            url, params=params, auth=HttpNtlmAuth(PI_USERNAME, PI_PASSWORD), verify=False
        )
        resp.raise_for_status()
        items = resp.json().get("Items", [])
        if not items:
            break
        all_items.extend(items)
        # move start to last timestamp + 1s to avoid overlaps
        last_timestamp = items[-1]["Timestamp"]
        last_time = pd.to_datetime(last_timestamp) + pd.Timedelta(seconds=1)
        current_start = last_time.isoformat()
        time.sleep(0.1)

    df = pd.DataFrame(
        [{"timestamp": it["Timestamp"], "value": it.get("Value")} for it in all_items]
    )
    return df
