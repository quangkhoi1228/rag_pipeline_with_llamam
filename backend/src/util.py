
import uuid


def generate_uuid():
    return str(uuid.uuid4())


def convert_int_to_string(value):
    """Chuyển đổi giá trị int sang str."""
    if isinstance(value, int):
        return str(value)
    return value  # Trả về giá trị như cũ nếu không phải là int
