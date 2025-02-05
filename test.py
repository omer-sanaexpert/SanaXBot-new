import requests
import json
import uuid
import os

# Zendesk Configuration
subdomain = os.environ.get("ZENDESK_SUBDOMAIN")  # Replace with your Zendesk subdomain
admin_email =os.environ.get("ZENDESK_EMAIL")  # Replace with your Zendesk agent email
api_token = os.environ.get("ZENDESK_API_TOKEN")  # Replace with your Zendesk API token

# API Setup
base_url = f"https://{subdomain}.zendesk.com"
auth = (f"{admin_email}/token", api_token)
headers = {"Content-Type": "application/json"}

# Create Temporary Anonymous User and Ticket
def create_anonymous_ticket():
    # Generate unique temporary identifier
    temp_email = f"anonymous_{uuid.uuid4().hex}@example.com"
    temp_name = "Test User AI Department"

    ticket_data = {
        "request": {
            "subject": "Support Request",
            "comment": {"body": "This is a test. Please ignore."},
            "requester": {
                "name": temp_name
            }
        }
    }

    response = requests.post(
        f"{base_url}/api/v2/requests",
        auth=auth,
        headers=headers,
        data=json.dumps(ticket_data)
    )

    if response.status_code == 201:
        ticket = response.json()["request"]
        print(f"Ticket created: {ticket['id']}")
        return ticket["requester_id"], ticket["id"]
    else:
        print(f"Error creating ticket: {response.text}")
        return None, None

# Update User Details
def update_user_details(requester_id, new_email, new_name):
    user_data = {
        "user": {
            "email": new_email,
            "name": new_name
        }
    }

    response = requests.put(
        f"{base_url}/api/v2/users/{requester_id}",
        auth=auth,
        headers=headers,
        data=json.dumps(user_data)
    )

    if response.status_code == 200:
        print("User updated successfully")
        return True
    else:
        print(f"Error updating user: {response.text}")
        return False

# Main Workflow
def main():
    # Step 1: Create anonymous ticket
    requester_id, ticket_id = create_anonymous_ticket()
    if not requester_id:
        return

    # Step 2: Collect user details (simulated input)
    print("\nPlease provide your contact information:")
    new_email = input("Email address: ").strip()
    new_name = input("Full name: ").strip()

    # Step 3: Update user details
    if update_user_details(requester_id, new_email, new_name):
        print(f"Ticket {ticket_id} now associated with {new_email}")
    else:
        print("Failed to update user details")

if __name__ == "__main__":
    main()