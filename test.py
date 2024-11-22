import requests
import time

WEBHOOK_URL = "https://discord.com/api/webhooks/1309356624582017127/IfKHtL5eOsg_faI3dOg24gg7iwiLVSuwF07tnCnJOY5h7PTXIs5I_-XF2nj_a1jPnhok"

while True:
    with open('/Users/ace/Documents/thesis/code/CLIPTrans/logs/job_name_1081145.err', 'r') as f:
        content = f.read()
    data = {
        "content": f"Updated content:\n{content}"
    }
    requests.post(WEBHOOK_URL, json=data)
    time.sleep(5)  # Adjust update interval as needed