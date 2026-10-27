"""Prints the latest version of a Secret Manager secret, authenticated as the machine's service account.

Used by vertex_entrypoint.sh inside training jobs, where gcloud isn't installed. Standard library only.
    python scripts/read_secret.py projects/<project>/secrets/<name>
"""
import base64
import json
import sys
import urllib.request

TOKEN_URL = "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token"


def read_secret(secret: str) -> str:
    token_request = urllib.request.Request(TOKEN_URL, headers={"Metadata-Flavor": "Google"})
    with urllib.request.urlopen(token_request, timeout=10) as response:
        token = json.load(response)["access_token"]

    secret_request = urllib.request.Request(f"https://secretmanager.googleapis.com/v1/{secret}/versions/latest:access",
                                            headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(secret_request, timeout=10) as response:
        return base64.b64decode(json.load(response)["payload"]["data"]).decode()


if __name__ == "__main__":
    print(read_secret(sys.argv[1]), end="")
