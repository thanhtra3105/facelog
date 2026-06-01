import time
import board
import busio
import adafruit_vl53l0x

i2c = busio.I2C(board.SCL, board.SDA)
vl53 = adafruit_vl53l0x.VL53L0X(i2c)

try:
    while True:
        t1 = time.time()
        distance = vl53.range
        t2 = time.time()

        print(f"Distance: {distance} mm | Read time: {(t2 - t1)*1000:.1f} ms")
        time.sleep(0.2)

except KeyboardInterrupt:
    print("\nStopped")
