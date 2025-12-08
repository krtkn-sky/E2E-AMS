import csv
import random
from datetime import datetime, timedelta

EVENT_TYPES = ["view", "click", "add_to_cart", "purchase"]

PRODUCT_IDS = list(range(1, 51))   # 50 products
USER_IDS = list(range(1001, 1021)) # 20 users

def generate_event():
    """Generate one realistic event."""
    event = {}

    # Random user and product
    event["user_id"] = random.choice(USER_IDS)
    event["product_id"] = random.choice(PRODUCT_IDS)

    # Random timestamp within last 7 days
    now = datetime.now()
    delta = timedelta(days=random.randint(0, 6), hours=random.randint(0,23), minutes=random.randint(0,59))
    event["timestamp"] = (now - delta).strftime("%Y-%m-%d %H:%M:%S")

    # Event logic: purchase only if clicked and/or added to cart recently
    rand_num = random.random()
    if rand_num < 0.5:
        event["event"] = "view"
    elif rand_num < 0.7:
        event["event"] = "click"
    elif rand_num < 0.9:
        event["event"] = "add_to_cart"
    else:
        event["event"] = "purchase"

    return event

def generate_events(n=10000):
    """Generate n events and save to CSV"""
    rows = [generate_event() for _ in range(n)]

    with open("data/events_raw.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "user_id", "product_id", "event"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated {n} events → data/events_raw.csv")

if __name__ == "__main__":
    generate_events()
