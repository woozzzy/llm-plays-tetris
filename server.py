import socket
import json
import time
import threading
import os

symbolic_keys = {
    "rotate180": 0x04,  # 'a'
    "hold": 0x06,  # 'c'
    "moveRight": 0x4F,  # Right Arrow
    "moveLeft": 0x50,  # Left Arrow
    "rotateCW": 0x1B,  # 'x'
    "rotateCCW": 0x1D,  # 'z'
    "hardDrop": 0x52,  # Up Arrow
    "softDrop": 0x51,  # Down Arrow
}

frame_time = 1.0 / 60.0


def process_events(events):
    scheduled_events = []
    for event in events:
        frame = event.get("frame", 0)
        subframe = event["data"].get("subframe", 0)
        scheduled_time = (frame + subframe) * frame_time
        scheduled_events.append((scheduled_time, event))

    scheduled_events.sort(key=lambda x: x[0])

    pressed_keys = set()
    start_time = time.monotonic()

    hid_device = open("/dev/hidg0", "rb+", buffering=0)

    for scheduled_time, event in scheduled_events:
        now = time.monotonic()
        wait_time = start_time + scheduled_time - now
        if wait_time > 0:
            time.sleep(wait_time)

        event_type = event["type"]
        key_name = event["data"].get("key")
        key_code = symbolic_keys.get(key_name)

        if key_code is None:
            print(f"Unknown key: {key_name}")
            continue

        if event_type == "keydown":
            pressed_keys.add(key_code)
        elif event_type == "keyup":
            pressed_keys.discard(key_code)
        else:
            continue

        report = [0x00] * 8
        key_codes = list(pressed_keys)[:6]
        for i, code in enumerate(key_codes):
            report[2 + i] = code

        hid_device.write(bytes(report))
        hid_device.flush()

    hid_device.write(b"\x00" * 8)
    hid_device.flush()
    hid_device.close()


def handle_client_connection(client_socket):
    data = b""
    while True:
        chunk = client_socket.recv(4096)
        if not chunk:
            break
        data += chunk
    client_socket.close()

    try:
        event_list = json.loads(data.decode("utf-8"))
        events = event_list.get("events", [])
        process_events(events)
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error: {e}")


def start_server():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("0.0.0.0", 12345))
    server.listen(1)
    print("Listening on port 12345...")
    while True:
        client_sock, address = server.accept()
        print(f"Accepted connection from {address}")
        client_handler = threading.Thread(target=handle_client_connection, args=(client_sock,))
        client_handler.start()


if __name__ == "__main__":
    start_server()
