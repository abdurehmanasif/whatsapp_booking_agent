import uuid
from fastapi import FastAPI, Request
from fastapi.responses import Response
from twilio.twiml.messaging_response import MessagingResponse
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, AIMessage
from langchain.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
import pandas as pd
import json
import os
import re
from datetime import datetime
from typing import Dict, Any
import logging
from dotenv import load_dotenv


# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Initialize Google Gemini
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.5)

# File paths
SERVICE_BOOKINGS_CSV = "service_bookings.csv"
TEST_DRIVE_BOOKINGS_CSV = "test_drive_bookings.csv"
USERS_CSV = "data/users.csv"
CONVERSATIONS_DIR = "conversations"
SERVICE_CENTERS_JSON = "data/service_centers.json"

# Ensure directories exist
os.makedirs(CONVERSATIONS_DIR, exist_ok=True)


# Initialize CSV files if they don't exist
def init_csv_files():
    booking_columns = [
        "user_id",
        "name",
        "city",
        "center_address",
        "car_model",
        "date",
        "time_slot",
        "booking_id",
        "status",
        "created_at",
        "updated_at",
    ]

    if not os.path.exists(SERVICE_BOOKINGS_CSV):
        pd.DataFrame(columns=booking_columns).to_csv(SERVICE_BOOKINGS_CSV, index=False)

    if not os.path.exists(TEST_DRIVE_BOOKINGS_CSV):
        pd.DataFrame(columns=booking_columns).to_csv(
            TEST_DRIVE_BOOKINGS_CSV, index=False
        )

    if not os.path.exists(USERS_CSV):
        # Sample users data
        users_data = [
            {
                "user_id": "1",
                "phone_number": "966512345678",
                "name": "Ahmed Al Saud",
                "car_model": "Lucid Air",
            },
            {
                "user_id": "2",
                "phone_number": "966598765432",
                "name": "Fatima Al Harbi",
                "car_model": "Lucid Gravity",
            },
            {
                "user_id": "3",
                "phone_number": "966577788899",
                "name": "Mohammed Al Qahtani",
                "car_model": "Lucid Air Grand Touring",
            },
            {
                "user_id": "4",
                "phone_number": "966511122233",
                "name": "Salman Al Otaibi",
                "car_model": "Lucid Air",
            },
            {
                "user_id": "5",
                "phone_number": "966566677788",
                "name": "Reem Al Shammari",
                "car_model": "Lucid Gravity",
            },
            {
                "user_id": "6",
                "phone_number": "966533344455",
                "name": "Abdullah Al Dossary",
                "car_model": "Lucid Air Grand Touring",
            },
            {
                "user_id": "7",
                "phone_number": "966544455566",
                "name": "Muna Al Zahrani",
                "car_model": "Lucid Air",
            },
        ]
        pd.DataFrame(users_data).to_csv(USERS_CSV, index=False)


init_csv_files()


# Utility functions
def get_user_info(user_id: Any) -> Dict[str, Any]:
    """Get user information from CSV"""
    try:
        user_id_str = _normalize_user_id(user_id)

        # Load as string to keep leading zeros, then normalise both id columns
        users_df = pd.read_csv(USERS_CSV, dtype=str)
        for col in ("user_id", "phone_number"):
            if col in users_df.columns:
                users_df[col] = users_df[col].apply(_normalize_user_id)

        # Match on either column
        user_row = users_df[
            (users_df["user_id"] == user_id_str)
            | (users_df["phone_number"] == user_id_str)
        ]

        if not user_row.empty:
            return user_row.iloc[0].to_dict()
    except Exception as e:
        logger.error(f"Error getting user info for {user_id}: {e}")
    return {
        "user_id": str(user_id),
        "phone_number": "Unknown",
        "name": "Customer",
        "car_model": "Unknown",
    }


def _normalize_user_id(user_id: Any) -> str:
    if user_id is None:
        return ""
    """
    Convert anything that represents a phone number
    (e.g. 'whatsapp:+923304641960', '+92 330 464 1960', 923304641960.0)
    into a clean digit-only string.
    """
    if user_id is None:
        return ""
    user_str = str(user_id)
    # Remove Twilio prefix if present
    user_str = user_str.replace("whatsapp:", "")
    # Keep digits only
    return re.sub(r"\D", "", user_str)


def load_conversation_history(user_id: Any) -> list:
    """Load conversation history from JSON"""
    file_path = os.path.join(CONVERSATIONS_DIR, f"{user_id}.json")
    if os.path.exists(file_path):
        try:
            with open(file_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading conversation history: {e}")
            return []
    return []


def save_conversation_history(user_id: Any, history: list):
    """Save conversation history to JSON"""
    file_path = os.path.join(CONVERSATIONS_DIR, f"{user_id}.json")
    try:
        with open(file_path, "w") as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving conversation: {e}")


def load_service_centers() -> list:
    """Load service centers from JSON"""
    with open(SERVICE_CENTERS_JSON, "r") as f:
        return json.load(f)


# Tools/Functions
@tool
def register_user(user_id: Any, name: str, car_model: str) -> str:
    """Register a new user"""
    user_id_str = _normalize_user_id(user_id)
    # Load or create the users DataFrame with columns as strings
    if os.path.exists(USERS_CSV):
        df = pd.read_csv(USERS_CSV, dtype=str)
    else:
        df = pd.DataFrame(columns=["user_id", "phone_number", "name", "car_model"])

    # Normalise ID columns for reliable comparison
    for col in ("user_id", "phone_number"):
        if col in df.columns:
            df[col] = df[col].apply(_normalize_user_id)

    # Check if the user already exists by ID or phone number
    if not df[
        (df["user_id"] == user_id_str) | (df.get("phone_number", "") == user_id_str)
    ].empty:
        return (
            "You are already registered. Would you like to book a service appointment?"
        )
    new_user = {
        "user_id": user_id_str,
        "phone_number": user_id_str,  # phone number is canonical ID
        "name": name.strip(),
        "car_model": car_model.strip(),
    }
    df = pd.concat([df, pd.DataFrame([new_user])], ignore_index=True)
    df.to_csv(USERS_CSV, index=False)
    return "You are now registered. Would you like to book a service appointment?"


@tool
def book_service_appointment(
    user_id: Any,
    name: str,
    city: str,
    center_address: str,
    car_model: str,
    date: str,
    time_slot: str,
) -> str:
    """Book a car service appointment"""
    try:
        # Validate inputs
        if not all([user_id, name, city, center_address, car_model, date, time_slot]):
            return "Missing required information. Please provide all details."

        # Convert user_id to string for consistency
        user_id_str = _normalize_user_id(user_id)

        df = pd.read_csv(SERVICE_BOOKINGS_CSV)

        # Convert user_id column to string for comparison
        if not df.empty:
            df["user_id"] = df["user_id"].apply(_normalize_user_id)

        # Check if user already has an active booking
        existing = df[(df["user_id"] == user_id_str) & (df["status"] != "cancelled")]
        if not existing.empty:
            return "You already have an active service appointment. Would you like to update or cancel it first?"

        # Validate date format
        try:
            datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            return "Invalid date format. Please use YYYY-MM-DD format."

        new_booking = {
            "user_id": user_id_str,
            "name": name.strip(),
            "city": city.strip(),
            "center_address": center_address.strip(),
            "car_model": car_model.strip(),
            "date": date,
            "time_slot": time_slot.strip(),
            "booking_id": str(uuid.uuid4()),
            "status": "pending",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }

        df = pd.concat([df, pd.DataFrame([new_booking])], ignore_index=True)
        df.to_csv(SERVICE_BOOKINGS_CSV, index=False)

        return f"✅ Service appointment booked successfully!\n📅 Date: {date}\n🕐 Time: {time_slot}\n📍 Location: {city}\n🏢 Center: {center_address}"

    except Exception as e:
        logger.error(f"Error booking service appointment: {e}")
        return "Sorry, there was an error booking your appointment. Please try again."


@tool
def get_service_centers(city: str) -> str:
    """Get service centers in a city"""
    service_centers = load_service_centers()
    centers = service_centers.get(city)
    if centers:
        result = f"Service centers in {city}:\n"
        for center in centers:
            result += f"• {center['name']}\n"
            result += f"  Address: {center['address']}\n"
            result += f"  Hours: {center['working_hours']}\n"
            if center["offer_test_drive"]:
                result += "  Test drives available: Yes\n"
            result += "\n"
        return result
    else:
        available_cities = list(service_centers.keys())
        return f"No service center found in {city}. Available cities: {', '.join(available_cities)}"


@tool
def get_cities_with_availability() -> str:
    """Get cities with available test drive slots - enhanced version"""
    try:
        service_centers = load_service_centers()
        result = "🏙️ **Available Cities for Test Drives:**\n\n"

        for city, centers in service_centers.items():
            result += f"📍 **{city}**\n"
            for center in centers:
                if center.get("offer_test_drive", False):
                    result += f"   • {center['name']}\n"
                    result += f"     📍 {center['address']}\n"

                    # Show available dates
                    available_dates = []
                    for timeslot_data in center.get("timeslots", []):
                        available_slots = [
                            slot for slot in timeslot_data["slots"] if slot["available"]
                        ]
                        if available_slots:
                            available_dates.append(timeslot_data["date"])

                    if available_dates:
                        result += (
                            f"     📅 Available dates: {', '.join(available_dates[:3])}"
                        )
                        if len(available_dates) > 3:
                            result += f" (and {len(available_dates) - 3} more)"
                        result += "\n"
                    else:
                        result += "     ❌ No available slots currently\n"
            result += "\n"

        return result
    except Exception as e:
        logger.error(f"Error getting cities with availability: {e}")
        return (
            "Sorry, there was an error retrieving available cities. Please try again."
        )


@tool
def get_available_timeslots(city: str, date: str) -> str:
    """Get available timeslots for a city on a given date"""
    try:
        service_centers = load_service_centers()
        city_clean = city.strip()

        if city_clean not in service_centers:
            available_cities = list(service_centers.keys())
            return f"❌ No service center found in {city_clean}.\n🏙️ Available cities: {', '.join(available_cities)}"

        centers = service_centers.get(city_clean, [])
        if not centers:
            return f"❌ No service centers available in {city_clean}"

        # Validate date format
        try:
            datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            return "❌ Invalid date format. Please use YYYY-MM-DD format (e.g., 2025-07-19)"

        result = f"📅 Available timeslots for {date} in {city_clean}:\n\n"
        has_available_slots = False

        for center in centers:
            result += f"🏢 **{center['name']}**\n"
            result += f"📍 {center['address']}\n"
            result += f"🕐 Hours: {center['working_hours']}\n"

            # Find matching date in timeslots
            date_found = False
            for timeslot_data in center.get("timeslots", []):
                if timeslot_data["date"] == date:
                    date_found = True
                    available_slots = [
                        slot for slot in timeslot_data["slots"] if slot["available"]
                    ]
                    if available_slots:
                        result += "✅ **Available times:**\n"
                        for slot in available_slots:
                            result += f"   • {slot['time']}\n"
                        has_available_slots = True
                    else:
                        result += "❌ No available slots for this date\n"
                    break

            if not date_found:
                result += f"❌ No timeslots available for {date}\n"
                # Show available dates
                available_dates = [ts["date"] for ts in center.get("timeslots", [])]
                if available_dates:
                    result += f"📅 Available dates: {', '.join(available_dates)}\n"

            result += "\n"

        if not has_available_slots:
            result += "💡 **Suggestion:** Try a different date or check available dates listed above."

        return result

    except Exception as e:
        logger.error(f"Error getting timeslots: {e}")
        return "Sorry, there was an error retrieving available timeslots. Please try again."


@tool
def book_test_drive(
    user_id: Any,
    name: str,
    city: str,
    center_address: str,
    car_model: str,
    date: str,
    time_slot: str,
) -> str:
    """Book a test drive appointment"""
    try:
        # Normalize user_id
        user_id_str = _normalize_user_id(user_id)

        df = pd.read_csv(TEST_DRIVE_BOOKINGS_CSV)

        # Normalize user_id column for comparison
        if not df.empty:
            df["user_id"] = df["user_id"].apply(_normalize_user_id)

        # Check if user already has a booking
        existing = df[(df["user_id"] == user_id_str) & (df["status"] != "cancelled")]
        if not existing.empty:
            return "You already have a test drive booked. Would you like to update it instead?"

        new_booking = {
            "user_id": user_id_str,
            "name": name,
            "city": city,
            "center_address": center_address,
            "car_model": car_model,
            "date": date,
            "time_slot": time_slot,
            "booking_id": str(uuid.uuid4()),
            "status": "pending",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }

        df = pd.concat([df, pd.DataFrame([new_booking])], ignore_index=True)
        df.to_csv(TEST_DRIVE_BOOKINGS_CSV, index=False)

        return f"Test drive booked successfully for {date} at {time_slot} in {city}."
    except Exception as e:
        logger.error(f"Error booking test drive: {e}")
        return "Sorry, there was an error booking your test drive. Please try again."


@tool
def update_service_appointment(
    user_id: Any,
    name: str = "",
    city: str = "",
    center_address: str = "",
    car_model: str = "",
    date: str = "",
    time_slot: str = "",
) -> str:
    """Update an existing service appointment"""
    try:
        user_id_str = _normalize_user_id(user_id)
        df = pd.read_csv(SERVICE_BOOKINGS_CSV)
        if not df.empty:
            df["user_id"] = df["user_id"].apply(_normalize_user_id)
        user_booking = df[df["user_id"] == user_id_str]

        if user_booking.empty:
            return "You don't have any service appointment to update. Would you like to book one?"

        # Update only provided fields
        updates = {}
        if name:
            updates["name"] = name
        if city:
            updates["city"] = city
        if center_address:
            updates["center_address"] = center_address
        if car_model:
            updates["car_model"] = car_model
        if date:
            updates["date"] = date
        if time_slot:
            updates["time_slot"] = time_slot

        if updates:
            updates["status"] = "pending"
            updates["updated_at"] = datetime.now().isoformat()
            for key, value in updates.items():
                # Use normalized user_id_str to ensure accurate row selection
                df.loc[df["user_id"] == user_id_str, key] = value

            df.to_csv(SERVICE_BOOKINGS_CSV, index=False)

            updated_fields = ", ".join(updates.keys())
            return f"Service appointment updated successfully. Updated fields: {updated_fields}"
        else:
            return "No updates provided. Please specify what you'd like to change."

    except Exception as e:
        logger.error(f"Error updating service appointment: {e}")
        return "Sorry, there was an error updating your appointment. Please try again."


@tool
def get_user_test_drive_bookings(user_id: Any) -> str:
    """Get user test drive bookings"""
    df = pd.read_csv(TEST_DRIVE_BOOKINGS_CSV)
    if not df.empty:
        df["user_id"] = df["user_id"].apply(_normalize_user_id)
    user_id_str = _normalize_user_id(user_id)
    user_bookings = df[df["user_id"] == user_id_str]

    if user_bookings.empty:
        return f"No test drive bookings found for user {user_id}."

    result = f"Test drive bookings for user {user_id}:\n"
    for _, booking in user_bookings.iterrows():
        result += f"• Booking ID: {booking['booking_id']}\n"
        result += f"  Name: {booking['name']}\n"
        result += f"  City: {booking['city']}\n"
        result += f"  Center: {booking['center_address']}\n"
        result += f"  Car Model: {booking['car_model']}\n"
        result += f"  Date: {booking['date']}\n"
        result += f"  Time: {booking['time_slot']}\n"
        result += f"  Status: {booking['status']}\n\n"
    return result


@tool
def get_user_service_bookings(user_id: Any) -> str:
    """Get user service bookings"""
    df = pd.read_csv(SERVICE_BOOKINGS_CSV)
    user_id_str = _normalize_user_id(user_id)
    if not df.empty:
        df["user_id"] = df["user_id"].apply(_normalize_user_id)
    user_bookings = df[df["user_id"] == user_id_str]

    if user_bookings.empty:
        return f"No service bookings found for user {user_id}."

    result = f"Service bookings for user {user_id}:\n"
    for _, booking in user_bookings.iterrows():
        result += f"• Booking ID: {booking['booking_id']}\n"
        result += f"  Name: {booking['name']}\n"
        result += f"  City: {booking['city']}\n"
        result += f"  Center: {booking['center_address']}\n"
        result += f"  Car Model: {booking['car_model']}\n"
        result += f"  Date: {booking['date']}\n"
        result += f"  Time: {booking['time_slot']}\n"
        result += f"  Status: {booking['status']}\n\n"
    return result


@tool
def cancel_test_drive(user_id: Any) -> str:
    """Cancel a test drive appointment (status -> cancelled)"""
    try:
        df = pd.read_csv(TEST_DRIVE_BOOKINGS_CSV)
        user_id_str = _normalize_user_id(user_id)
        if not df.empty:
            df["user_id"] = df["user_id"].apply(_normalize_user_id)
        user_bookings = df[df["user_id"] == user_id_str]
        if user_bookings.empty:
            return "You don't have any test drive to cancel."
        else:
            # Use normalized user_id_str to ensure accurate row selection
            df.loc[df["user_id"] == user_id_str, "status"] = "cancelled"
            df.to_csv(TEST_DRIVE_BOOKINGS_CSV, index=False)
            return "Test drive cancelled successfully."
    except Exception as e:
        logger.error(f"Error cancelling test drive: {e}")
        return "Sorry, there was an error cancelling your test drive. Please try again."


@tool
def cancel_service_appointment(user_id: Any) -> str:
    """Cancel a service appointment"""
    try:
        user_id_str = _normalize_user_id(user_id)
        df = pd.read_csv(SERVICE_BOOKINGS_CSV)
        if not df.empty:
            df["user_id"] = df["user_id"].apply(_normalize_user_id)

        user_bookings = df[
            (df["user_id"] == user_id_str) & (df["status"] != "cancelled")
        ]

        if user_bookings.empty:
            return "❌ You don't have any active service appointments to cancel."

        # Cancel the booking
        df.loc[
            (df["user_id"] == user_id_str) & (df["status"] != "cancelled"), "status"
        ] = "cancelled"
        df.loc[
            (df["user_id"] == user_id_str) & (df["status"] == "cancelled"), "updated_at"
        ] = datetime.now().isoformat()
        df.to_csv(SERVICE_BOOKINGS_CSV, index=False)

        return "✅ Service appointment cancelled successfully."

    except Exception as e:
        logger.error(f"Error cancelling service appointment: {e}")
        return (
            "Sorry, there was an error cancelling your appointment. Please try again."
        )


@tool
def update_test_drive(
    user_id: Any,
    name: str = "",
    city: str = "",
    center_address: str = "",
    car_model: str = "",
    date: str = "",
    time_slot: str = "",
) -> str:
    """Update an existing test drive appointment"""
    try:
        df = pd.read_csv(TEST_DRIVE_BOOKINGS_CSV)
        # Ensure consistent string comparison for user_id
        if not df.empty:
            df["user_id"] = df["user_id"].apply(_normalize_user_id)

        user_id_str = _normalize_user_id(user_id)
        user_booking = df[df["user_id"] == user_id_str]

        if user_booking.empty:
            return (
                "You don't have any test drive to update. Would you like to book one?"
            )

        # Update only provided fields
        updates = {}
        if name:
            updates["name"] = name
        if city:
            updates["city"] = city
        if center_address:
            updates["center_address"] = center_address
        if car_model:
            updates["car_model"] = car_model
        if date:
            updates["date"] = date
        if time_slot:
            updates["time_slot"] = time_slot

        if updates:
            updates["status"] = "pending"
            updates["updated_at"] = datetime.now().isoformat()
            for key, value in updates.items():
                # Use normalized user_id_str to ensure accurate row selection
                df.loc[df["user_id"] == user_id_str, key] = value

            df.to_csv(TEST_DRIVE_BOOKINGS_CSV, index=False)

            updated_fields = ", ".join(updates.keys())
            return f"Test drive updated successfully. Updated fields: {updated_fields}"
        else:
            return "No updates provided. Please specify what you'd like to change."

    except Exception as e:
        logger.error(f"Error updating test drive: {e}")
        return "Sorry, there was an error updating your test drive. Please try again."


# Agent setup
tools = [
    book_service_appointment,
    book_test_drive,
    update_service_appointment,
    update_test_drive,
    get_service_centers,
    get_user_test_drive_bookings,
    get_user_service_bookings,
    cancel_test_drive,
    cancel_service_appointment,
    get_available_timeslots,
    get_cities_with_availability,
    register_user,
]

system_prompt = """You are Lucid Motors' Middle East (Saudi Arabia) Customer Service Agent. You can help users:
1. Book car service appointments
2. Book test drive appointments  
3. Update or cancel existing bookings
4. View booking details and status

**PERSONALITY & TONE:**
- Be warm, professional, and helpful
- Use appropriate emojis for better engagement
- Address users by their first name when possible and tell them what car they have
- Be conversational but efficient
- Mimic a human agent, not a robot
- If user query is in Arabic, respond in Arabic

**WORKFLOW GUIDELINES:**
When user wants to book a service appointment or test drive:
1. Greet them warmly and acknowledge their interest
2. IMMEDIATELY use get_cities_with_availability to show all available cities and dates
3. Let user choose their preferred city
4. Once city is selected, automatically get available timeslots for their preferred date
5. Show them all available options before asking them to choose
6. Confirm all details before booking

When user wants to update a booking (service or test drive):
1. FIRST check their current bookings using get_user_test_drive_bookings or get_user_service_bookings
2. Show them their current booking details
3. Ask what they want to change (city, date, time)
4. Provide available options for the change they want to make
5. Update the booking with confirmation

**CRITICAL RULES:**
- NEVER ask users to provide information you can get from tools
- ALWAYS show available options BEFORE asking users to choose
- ALWAYS use get_cities_with_availability for initial city selection
- ALWAYS convert dates to YYYY-MM-DD format (assume 2025 if year not specified)
- ALWAYS validate information before booking
- If timeslots are unavailable, suggest alternatives immediately
- When updating, ALWAYS show current booking first

**REGISTRATION RULES:**
• `user_id` **is the caller's WhatsApp phone number (digits only); it is the same value as `phone_number`.** Treat them interchangeably.
• If the user is not found in the database (i.e. not registered), complete the requested booking / update flow first.
• At the END of that flow politely ask: "Would you like to register so we can serve you faster next time?".
• If the user says **YES**, use the information you already have (name, car model, phone number) and call `register_user`.
• Ask the user for their name and car model if not provided.(Customer is not a valid name)
• If any required field is still missing, ask specifically for it before calling `register_user`.
• Confirm successful registration back to the user.

**REQUIRED INFORMATION FOR BOOKINGS:**
All booking tools require these parameters:
- user_id (from user profile)
- name (from user profile)  
- city (from available cities)
- center_address (from service centers in selected city)
- car_model (from user profile or ask if missing)
- date (YYYY-MM-DD format)
- time_slot (from available timeslots)

**ERROR HANDLING:**
- If a booking fails, explain why and offer alternatives
- If information is missing, ask for it specifically
- If dates/times are unavailable, show available options
- Always be helpful and solution-oriented

**TOOLS AVAILABLE:**
- get_cities_with_availability: Get all available cities with test drive availability
- get_service_centers: Get centers in a specific city
- get_available_timeslots: Get available slots for date/city
- book_service_appointment: Book service appointment
- book_test_drive: Book test drive
- update_service_appointment: Update existing service booking
- update_test_drive: Update existing test drive
- cancel_service_appointment: Cancel service appointment
- cancel_test_drive: Cancel test drive
- get_user_service_bookings: View user's service bookings
- get_user_test_drive_bookings: View user's test drive bookings
- register_user: Register a new user

**CONFIRMATION PROCESS:**
Before booking, always confirm:
"Let me confirm your booking details:
👤 Name: [name]
🚗 Car: [car_model]  
📍 City: [city]
🏢 Center: [center_name]
📅 Date: [date]
🕐 Time: [time_slot]

Is this correct? Type 'YES' to confirm or let me know what to change."

Remember: Lucid Motors offers premium electric vehicles (Lucid Air, Lucid Air Grand Touring, Lucid Gravity) with exceptional service experience.
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


def process_message(user_id: Any, message: str) -> str:
    """Process incoming message and return response"""
    try:
        # Get user info for personalization
        user_info = get_user_info(user_id)

        # Load conversation history
        history = load_conversation_history(user_id)

        # Convert history to LangChain format
        chat_history = []
        for msg in history:
            if msg["type"] == "human":
                chat_history.append(HumanMessage(content=msg["content"]))
            else:
                chat_history.append(AIMessage(content=msg["content"]))

        # Add user context to the message
        contextualized_message = f"Here is the User Profile to personalise your responses->\n User ID: {user_id}(used to book the service/test drive appointment), User Name: {user_info['name']}(used to greet the user), Car Details: {user_info['car_model']})\nMessage: {message}"
        # Get agent response
        response = agent_executor.invoke(
            {"input": contextualized_message, "chat_history": chat_history}
        )

        # Update conversation history
        history.append(
            {
                "type": "human",
                "content": message,
                "timestamp": datetime.now().isoformat(),
            }
        )
        history.append(
            {
                "type": "ai",
                "content": response["output"],
                "timestamp": datetime.now().isoformat(),
            }
        )

        # Keep only last 20 messages to manage memory
        if len(history) > 50:  # Instead of 20
            history = history[-30:]  # Keep more recent context

        save_conversation_history(user_id, history)

        return response["output"]

    except Exception as e:
        logger.error(f"Error processing message: {e}")
        return "Sorry, I encountered an error. Please try again."


@app.post("/webhook")
async def webhook(request: Request):
    """Handle incoming WhatsApp messages"""
    try:
        form_data = await request.form()

        # Extract message details
        message_body = form_data.get("Body", "").strip()
        from_number = form_data.get("From", "")

        # Extract user ID from phone number (remove whatsapp: prefix)
        user_id = _normalize_user_id(from_number)

        if not message_body:
            return Response(content="", media_type="text/plain")

        # Process message with agent
        response_text = process_message(user_id, message_body)

        # Create Twilio response
        twiml_response = MessagingResponse()
        twiml_response.message(response_text)

        return Response(content=str(twiml_response), media_type="text/xml")

    except Exception as e:
        logger.error(f"Webhook error: {e}")
        twiml_response = MessagingResponse()
        twiml_response.message(
            "Sorry, I'm having trouble right now. Please try again later."
        )
        return Response(content=str(twiml_response), media_type="text/xml")


@app.get("/")
async def root():
    return {"message": "WhatsApp Car Booking Bot is running!"}


@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}
