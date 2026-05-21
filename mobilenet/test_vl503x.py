import time
import board
import busio
import adafruit_vl53l0x

# Khởi tạo giao tiếp I2C (sử dụng chân mặc định SCL và SDA của Pi)
i2c = busio.I2C(board.SCL, board.SDA)

# Khởi tạo cảm biến
vl53 = adafruit_vl53l0x.VL53L0X(i2c)

print("Đang khởi động cảm biến VL53L0X...")
print("Nhấn Ctrl+C để thoát chương trình.\n")

try:
    while True:
        # Lấy giá trị khoảng cách tính bằng mm
        distance = vl53.range
        
        print("Khoảng cách: {} mm".format(distance))
        
        # Đợi 0.1 giây trước khi đo lần tiếp theo
        time.sleep(0.1)

except KeyboardInterrupt:
    # Xử lý khi người dùng nhấn Ctrl+C
    print("\nĐã dừng chương trình đo.")