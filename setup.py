#!/usr/bin/env python3
"""
Quick setup script for the WhatsApp Car Booking Bot
"""
import os
import pandas as pd
from datetime import datetime


def create_sample_data():
    """Create sample CSV files for testing"""

    # Create sample users
    users_data = [
        {
            "user_id": "whatsapp:+1234567890",
            "name": "John Doe",
            "car_model": "Honda Civic",
        },
        {
            "user_id": "whatsapp:+0987654321",
            "name": "Jane Smith",
            "car_model": "Toyota Camry",
        },
        {
            "user_id": "whatsapp:+1122334455",
            "name": "Bob Wilson",
            "car_model": "Ford Focus",
        },
    ]

    users_df = pd.DataFrame(users_data)
    users_df.to_csv("users.csv", index=False)
    print("✓ Created users.csv with sample data")

    # Create empty booking CSV files
    booking_columns = [
        "user_id",
        "city",
        "center_address",
        "date",
        "time_slot",
        "created_at",
        "updated_at",
    ]

    empty_df = pd.DataFrame(columns=booking_columns)
    empty_df.to_csv("service_bookings.csv", index=False)
    print("✓ Created service_bookings.csv")

    empty_df.to_csv("test_drive_bookings.csv", index=False)
    print("✓ Created test_drive_bookings.csv")

    # Create conversations directory
    os.makedirs("conversations", exist_ok=True)
    print("✓ Created conversations directory")


def create_env_file():
    """Create .env file from template"""
    if not os.path.exists(".env"):
        with open(".env.example", "r") as f:
            template = f.read()

        with open(".env", "w") as f:
            f.write(template)

        print("✓ Created .env file from template")
        print("⚠️  Please update .env with your actual API keys and credentials")
    else:
        print("⚠️  .env file already exists")
    print("⚠️  Get your Google API key from: https://ai.google.dev/")
    print("⚠️  Make sure to enable the Gemini API in Google Cloud Console")
    print("⚠️  Make sure to enable the WhatsApp API in Twilio Console")


def main():
    print("Setting up WhatsApp Car Booking Bot...")
    print("=" * 50)

    create_sample_data()
    create_env_file()

    print("\nSetup complete! 🎉")
    print("\nNext steps:")
    print("1. Update .env with your Google API key and Twilio credentials")
    print("2. Get Google API key from: https://ai.google.dev/")
    print("3. Install dependencies: pip install -r requirements.txt")
    print("4. Run the bot: python main.py")
    print("5. Set up ngrok for local testing: ngrok http 8000")
    print("6. Update Twilio webhook URL with your ngrok URL + /webhook")


if __name__ == "__main__":
    main()
