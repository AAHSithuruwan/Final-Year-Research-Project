import os
import time
import requests


# Create GitHub API request headers
# The GITHUB_TOKEN environment variable is used for authentication
# If the token is not set, the rate limit will be stricter
def create_github_headers():
    github_token = os.getenv("GITHUB_TOKEN")

    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2026-03-10"
    }

    if github_token:
        headers["Authorization"] = f"Bearer {github_token}"
    else:
        print("\nWarning: GITHUB_TOKEN environment variable is not set.")
        print("GitHub API rate limits will be stricter without authentication.")

    return headers


# Extract GitHub rate limit details from the response headers
def get_rate_limit_details(response):
    return {
        "limit": response.headers.get("X-RateLimit-Limit"),
        "remaining": response.headers.get("X-RateLimit-Remaining"),
        "reset": response.headers.get("X-RateLimit-Reset"),
        "resource": response.headers.get("X-RateLimit-Resource")
    }


# Wait for the GitHub API rate limit to reset if it has been reached
def wait_for_rate_limit_reset(response):
    rate_limit_details = get_rate_limit_details(response)

    remaining = rate_limit_details["remaining"]
    reset_time = rate_limit_details["reset"]

    if remaining == "0" and reset_time is not None:
        reset_timestamp = int(reset_time)
        current_timestamp = int(time.time())
        wait_seconds = max(reset_timestamp - current_timestamp + 5, 0)

        print(f"\nGitHub API rate limit reached. Waiting {wait_seconds} seconds...")
        time.sleep(wait_seconds)


# Wait before retrying a failed GitHub API request
def wait_before_retry_failed_request(attempt, base_wait_seconds=10):
    wait_seconds = base_wait_seconds * attempt

    print(f"Waiting {wait_seconds} seconds before retry...")
    time.sleep(wait_seconds)


# Call the GitHub API with retry and rate-limit handling
def call_github_api(url, headers, params=None, max_retries=3):
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(
                url,
                headers=headers,
                params=params,
                timeout=30
            )

        except requests.exceptions.RequestException as error:
            print(f"\nGitHub API request error: {error}")

            if attempt < max_retries:
                wait_before_retry_failed_request(attempt)
                continue

            return None

        if response.status_code == 200:
            return response.json()

        if response.status_code in [403, 429]:
            wait_for_rate_limit_reset(response)

            if attempt < max_retries:
                wait_before_retry_failed_request(attempt)
                continue

        if response.status_code >= 500:
            print(f"\nGitHub server error: {response.status_code}")

            if attempt < max_retries:
                wait_before_retry_failed_request(attempt)
                continue

        print("\nGitHub API request failed.")
        print(f"Status code: {response.status_code}")
        print(f"Response: {response.text}")

        return None

    return None