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
        user_id_str = str(user_id).strip()
        users_df = pd.read_csv(USERS_CSV)
        users_df["user_id"] = users_df["user_id"].astype(str)
        user_row = users_df[users_df["user_id"] == user_id_str]

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
        df = pd.read_csv(SERVICE_BOOKINGS_CSV)

        # Check if user already has a booking
        existing = df[df["user_id"] == user_id]
        if not existing.empty:
            return "You already have a service appointment booked. Would you like to update it instead?"

        new_booking = {
            "user_id": user_id,
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
        df.to_csv(SERVICE_BOOKINGS_CSV, index=False)

        return f"Service appointment booked successfully for {date} at {time_slot} in {city}."
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
def get_cities() -> str:
    """Get all cities"""
    service_centers = load_service_centers()
    cities = list(service_centers.keys())
    return f"Cities with Lucid Motors service centers: {', '.join(cities)}"


@tool
def get_available_timeslots(city: str, date: str) -> str:
    """Get available timeslots for a city on a given date"""
    service_centers = load_service_centers()
    if city not in service_centers:
        return f"No service center found in {city}"

    centers = service_centers.get(city)
    if not centers:
        return f"No service center found in {city}"

    result = f"Available timeslots for {date} in {city}:\n"

    for center in centers:
        result += f"\n{center['name']}:\n"

        # Find matching date in timeslots
        date_found = False
        for timeslot_data in center.get("timeslots", []):
            if timeslot_data["date"] == date:
                date_found = True
                available_slots = [
                    slot for slot in timeslot_data["slots"] if slot["available"]
                ]
                if available_slots:
                    result += "  Available times:\n"
                    for slot in available_slots:
                        result += f"  • {slot['time']}\n"
                else:
                    result += "  No available slots for this date\n"
                break

        if not date_found:
            result += f"  No timeslots available for {date}\n"
            # Show available dates
            available_dates = [ts["date"] for ts in center.get("timeslots", [])]
            if available_dates:
                result += f"  Available dates: {', '.join(available_dates)}\n"

    return result


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
        df = pd.read_csv(TEST_DRIVE_BOOKINGS_CSV)

        # Check if user already has a booking
        existing = df[df["user_id"] == user_id]
        if not existing.empty:
            return "You already have a test drive booked. Would you like to update it instead?"

        new_booking = {
            "user_id": user_id,
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
        df = pd.read_csv(SERVICE_BOOKINGS_CSV)
        user_booking = df[df["user_id"] == user_id]

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
                df.loc[df["user_id"] == user_id, key] = value

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
    user_bookings = df[df["user_id"] == user_id]

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
    user_bookings = df[df["user_id"] == user_id]

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
        user_bookings = df[df["user_id"] == user_id]
        if user_bookings.empty:
            return "You don't have any test drive to cancel."
        else:
            df.loc[df["user_id"] == user_id, "status"] = "cancelled"
            df.to_csv(TEST_DRIVE_BOOKINGS_CSV, index=False)
            return "Test drive cancelled successfully."
    except Exception as e:
        logger.error(f"Error cancelling test drive: {e}")
        return "Sorry, there was an error cancelling your test drive. Please try again."


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
        user_booking = df[df["user_id"] == user_id]

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
                df.loc[df["user_id"] == user_id, key] = value

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
    get_available_timeslots,
    get_cities,
]

system_prompt = """You are Lucid Motors' Middle East (Saudi Arabia) Customer Service Agent. You can help users:
1. Book car service appointments
2. Book test drive appointments  
3. Updating or cancelling existing bookings
4. Viewing or resending booking details

Guidelines:
- Always use the user's name and car model after getting the user profile
- Be conversational and friendly
- If user greets you, greet them back appropriately and ask them how you can help them.
- Ask for required information step by step: city, center address, date, time slot
- Always confirm all details before booking
- For updates, ask what they want to change
- If user has existing bookings, inform them appropriately
- Keep responses concise and helpful
- Never ask the user to tell them city, car model, date, timeslot, center address from themselves, always use the tools to get the options and ask them to choose where you can.
- If user asks about Lucid Motors, introduce yourself as the customer service agent and tell them that you are here to help them with their bookings
- Lucid Motors is a car company that sells electric cars, currently selling the Lucid Air, Lucid Air Grand Touring and Lucid Gravity.

IMPORTANT DATE HANDLING:
- When user provides a date like "19th july", convert it to YYYY-MM-DD format (e.g., "2025-07-19")
- Always assume current year is 2025 unless user specifies otherwise
- Use get_available_timeslots to check what dates and times are actually available
- If requested date has no slots, suggest available alternatives

You have access to the following tools:
- book_service_appointment: Book a car service appointment
- book_test_drive: Book a test drive appointment.
- update_service_appointment: Update an existing service appointment.
- update_test_drive: Update an existing test drive appointment.
- get_service_centers: Get service centers in a city
- get_user_test_drive_bookings: Get user's existing test drive bookings
- get_user_service_bookings: Get user's existing service bookings
- cancel_test_drive: Cancel a test drive appointment
- get_available_timeslots: Get available timeslots for a service center on a given date
- get_cities: Get all cities

Required information for booking:
- City (always use get_cities to get the list of cities with service centers/test drive centers)
- Car model (if not provided in the user profile, ask for it)
- Service/Test Drive Center (always use get_service_centers to get the list of service centers in a city)
- Center address (always use get_service_centers to get the list of service centers in a city)
- Date (format: YYYY-MM-DD) (use get_available_timeslots to get the list of available timeslots for a service center on a given date)
- Time slot (format: HH:MM AM/PM) (use get_available_timeslots to get the list of available timeslots for a service center on a given date)

All bookings/updates tools require the following information:
- user_id (provider in the user profile)
- name (provider in the user profile)
- city (always use get_cities to get the list of cities with service centers/test drive centers)
- center_address (always use get_service_centers to get the list of service centers in a city)
- car_model (if not provided in the user profile, ask for it)
- date (format: YYYY-MM-DD) (use get_available_timeslots to get the list of available timeslots for a service center on a given date)
- time_slot (format: HH:MM AM/PM) (use get_available_timeslots to get the list of available timeslots for a service center on a given date)
- booking_id (generated by the tool)
- status (pending, cancelled, completed)

Always confirm all details before calling the booking functions.
📌 Reminders:
- Don't assume inputs. Always confirm.
- If updating, ask clearly what user wants to change.
- If user has existing bookings, offer to update or cancel.
- If timeslot is full, show alternate times.
- Parse dates correctly (assume 2025 if year not specified)

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
        user_id = from_number.replace("whatsapp:", "")

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
