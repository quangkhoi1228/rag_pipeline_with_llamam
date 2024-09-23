import uuid
from datetime import datetime

from fastapi import APIRouter, HTTPException
from src.util import generate_uuid
from src.entity import Room
from src.database import pg_create_connection

router = APIRouter()

# Pydantic model for Room data


# Function to generate a unique UUID


# CRUD operations

@router.post("/", response_model=Room)
async def create_room(room: Room):
    conn, cur = pg_create_connection()
    try:
        # Generate a unique UUID for the room ID
        room.id = generate_uuid()

        # Insert the room data into the database
        cur.execute(
            "INSERT INTO room (id, name, created_date) VALUES (%s, %s, %s)",
            (room.id, room.name, room.created_date),
        )
        conn.commit()

        return room
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error creating room: {e}")
    finally:
        cur.close()
        conn.close()


@router.get("/", response_model=list[Room])
async def get_all_rooms():
    conn, cur = pg_create_connection()
    try:
        cur.execute("SELECT * FROM room")
        rooms_data = cur.fetchall()

        rooms = [Room(**room_data) for room_data in rooms_data]
        return rooms
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error getting all rooms: {e}")
    finally:
        cur.close()
        conn.close()


@router.get("/{room_id}", response_model=Room)
async def get_room(room_id: str):
    conn, cur = pg_create_connection()
    try:
        cur.execute("SELECT * FROM room WHERE id = %s", (room_id,))
        room_data = cur.fetchone()

        if room_data is None:
            raise HTTPException(status_code=404, detail="Room not found")

        room = Room(**room_data)
        return room
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting room: {e}")
    finally:
        cur.close()
        conn.close()


@router.put("/{room_id}", response_model=Room)
async def update_room(room_id: str, room: Room):
    conn, cur = pg_create_connection()
    try:
        cur.execute(
            "UPDATE room SET name = %s WHERE id = %s", (room.name, room_id)
        )
        conn.commit()

        cur.execute("SELECT * FROM room WHERE id = %s", (room_id,))
        updated_room_data = cur.fetchone()

        if updated_room_data is None:
            raise HTTPException(status_code=404, detail="Room not found")

        updated_room = Room(**updated_room_data)
        return updated_room
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error updating room: {e}")
    finally:
        cur.close()
        conn.close()


@router.delete("/{room_id}")
async def delete_room(room_id: str):
    conn, cur = pg_create_connection()
    try:
        cur.execute("DELETE FROM room WHERE id = %s", (room_id,))
        conn.commit()

        if cur.rowcount == 0:
            raise HTTPException(status_code=404, detail="Room not found")

        return {"message": "Room deleted successfully"}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error deleting room: {e}")
    finally:
        cur.close()
        conn.close()
