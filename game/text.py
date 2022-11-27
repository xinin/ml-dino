from datetime import datetime

dt = datetime.now()

ts = datetime.timestamp(dt)

print("Date and time is:", dt)
print("Timestamp is:", int(ts))