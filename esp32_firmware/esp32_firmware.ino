/*
 * BeeGuard ESP32 Edge Firmware
 * ============================
 * 2× MG996R servos (pan/tilt)
 * DHT11 (temperature, humidity)
 * HC-SR05 (water tank level)
 * 4× 50kg load cell + HX711 (honey weight)
 * Water pump via 5V opto-isolated relay (defend against hornet attacks)
 * Serial USB (115200 baud) to Python host.
 *
 * Wiring:
 *   Pan  Servo (Shoulder) → GPIO 26
 *   Tilt Servo (Elbow)    → GPIO 25
 *   Pump Relay IN         → GPIO 27   (active HIGH — xem PUMP_ACTIVE_LOW)
 *   DHT11 Data            → GPIO 21   (module 3 chân thường có sẵn pull-up; nếu DHT11 rời thì gắn 4.7k–10k giữa DATA và VCC)
 *   HC-SR05 TRIG / ECHO   → GPIO 18 / 19
 *   HX711 DT / SCK        → GPIO 32 / 33
 *
 *   ⚠ Bơm phải có NGUỒN RIÊNG (không cấp qua ESP32). Relay chỉ đóng/cắt
 *     dòng cho bơm. Chia GND chung giữa ESP32 và nguồn bơm.
 *
 * Required libraries (Arduino IDE → Library Manager):
 *   - ESP32Servo
 *   - DHT sensor library (Adafruit)
 *   - HX711_ADC by Olav Kallhovd      <-- KHÔNG phải HX711 của Bogdan/Rob Tillaart
 *
 * Serial protocol:
 *   Host → ESP32:
 *     "PAN,TILT\n"     servo angles, e.g. "90,45\n"
 *     "HOME\n"         center both servos
 *     "PUMP,FIRE\n"    bắn nước 5s rồi tự tắt + cooldown 5s mới bắn tiếp
 *     "TARE\n"         tare load cell (non-blocking); offset mới được LƯU vào NVS
 *                      → lần boot sau khôi phục gốc 0 này, không tare lại.
 *     "PING\n"         responds "PONG\n"
 *
 *   ESP32 → Host (every 5s):
 *     "SENSOR:temp,humidity,distance_cm,weight_kg\n"
 *     -999.0 = sensor error / not ready.
 *   ESP32 → Host (pump events):
 *     "OK:PUMP_FIRE"      bắt đầu bắn
 *     "BUSY:PUMP"         đang FIRING hoặc COOLDOWN — fire bị bỏ qua
 *     "OK:PUMP_STOP"      hết 5s bắn → tắt relay, vào cooldown
 *     "OK:PUMP_READY"     hết cooldown → IDLE, sẵn sàng bắn tiếp
 *   ESP32 → Host (load cell events lúc boot):
 *     "OK:TARE_RESTORED,<offset>"  có offset trong NVS → khôi phục, không tare
 *     "OK:TARE_INITIAL"            lần đầu boot (NVS rỗng) → tare cứng + lưu offset
 *     "OK:TARE_COMPLETE"           tare từ host hoàn tất → offset đã lưu vào NVS
 */

#include <ESP32Servo.h>
#include <DHT.h>
#include <HX711_ADC.h>
#include <Preferences.h>

// ============== PIN DEFINITIONS ==============
#define PAN_SERVO_PIN   26
#define TILT_SERVO_PIN  25
#define PUMP_PIN        27

// Module relay 5V active-HIGH (kích ở mức HIGH).
// Lý do dùng active-HIGH: khi ESP32 boot/reset, GPIO ở trạng thái high-Z ~300-500ms.
// Active-HIGH + pull-down (nội hoặc ngoài) → mức OFF mặc định, relay không tự đóng khi cắm điện hay mở COM.
#define PUMP_ACTIVE_LOW false

#define PUMP_FIRE_MS      5000UL   // thời gian bắn nước
#define PUMP_COOLDOWN_MS  5000UL   // thời gian nghỉ trước lần bắn kế

#define DHT_PIN         21
#define DHT_TYPE        DHT11

#define TRIG_PIN        18
#define ECHO_PIN        19

#define HX711_DT_PIN    32
#define HX711_SCK_PIN   33

// ============== LOAD CELL CALIBRATION ==============
// Sau khi cân vật chuẩn: new_factor = old_factor * (reading / true_weight_kg)
// Đã calibrate với chai 1 kg → reading ~63 kg ở factor 420 → factor mới = 420 * 63 = 26460.
#define LOADCELL_CAL_FACTOR  26460.0f

// Thời gian chờ HX711 ổn định khi khởi động (ms).
#define LOADCELL_STABILIZE_MS 2000

// ============== SERVO CONFIGURATION ==============
#define PAN_MIN         0
#define PAN_MAX         180
#define TILT_MIN        0
#define TILT_MAX        180
#define PAN_HOME        90
#define TILT_HOME       90

#define SERVO_MIN_US    500
#define SERVO_MAX_US    2500

#define SMOOTH_FACTOR   0.3f

// ============== TIMING ==============
#define SENSOR_INTERVAL 5000   // ms

// ============== OBJECTS ==============
Servo panServo;
Servo tiltServo;
DHT dht(DHT_PIN, DHT_TYPE);
HX711_ADC LoadCell(HX711_DT_PIN, HX711_SCK_PIN);
Preferences prefs;

// NVS lưu tare offset → lần boot sau dùng lại gốc 0 đã calibrate, không cần tare lại.
// Sentinel để phát hiện key chưa tồn tại (lần boot đầu tiên).
const char* PREFS_NS       = "beeguard";
const char* PREFS_KEY_TARE = "tare_off";
const long  TARE_SENTINEL  = 0x7FFFFFFFL;

float currentPan  = PAN_HOME;
float currentTilt = TILT_HOME;
float targetPan   = PAN_HOME;
float targetTilt  = TILT_HOME;

String inputBuffer = "";

unsigned long lastUpdateTime = 0;
const unsigned long UPDATE_INTERVAL = 10;

unsigned long lastSensorTime = 0;
bool loadCellReady = false;

// ============== PUMP STATE MACHINE ==============
enum PumpState { PUMP_IDLE, PUMP_FIRING, PUMP_COOLDOWN };
PumpState pumpState = PUMP_IDLE;
unsigned long pumpStateEntered = 0;

inline void pumpRelayOn()  { digitalWrite(PUMP_PIN, PUMP_ACTIVE_LOW ? LOW  : HIGH); }
inline void pumpRelayOff() { digitalWrite(PUMP_PIN, PUMP_ACTIVE_LOW ? HIGH : LOW ); }

// ============== SETUP ==============
void setup() {
  Serial.begin(115200);
  Serial.setTimeout(10);

  // Boot an toàn cho relay Active HIGH:
  // 1. INPUT_PULLDOWN trước → chân được kéo LOW bằng pull-down nội ngay khi setup chạy,
  //    relay nhả (chứ không bị kích bởi nhiễu trên dây IN khi pin còn high-Z).
  // 2. Đặt latch LOW (mức OFF cho active-HIGH).
  // 3. Chuyển sang OUTPUT → vẫn LOW → relay vẫn nhả.
  // 4. Gọi pumpRelayOff() cho chắc (nếu polarity flag thay đổi).
  if (PUMP_ACTIVE_LOW) {
    pinMode(PUMP_PIN, INPUT_PULLUP);
  } else {
    pinMode(PUMP_PIN, INPUT_PULLDOWN);
  }
  digitalWrite(PUMP_PIN, PUMP_ACTIVE_LOW ? HIGH : LOW);
  pinMode(PUMP_PIN, OUTPUT);
  pumpRelayOff();

  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);
  digitalWrite(TRIG_PIN, LOW);

  dht.begin();

  // HX711_ADC: chỉ ổn định trong setup (không tare ở đây), sau đó loop chỉ gọi update().
  // Tare offset được khôi phục từ NVS để có thể đặt sẵn vật lên cân lúc khởi động.
  prefs.begin(PREFS_NS, false);
  LoadCell.begin();
  LoadCell.start(LOADCELL_STABILIZE_MS, false /*no tare*/);
  if (LoadCell.getTareTimeoutFlag() || LoadCell.getSignalTimeoutFlag()) {
    Serial.println("ERR:HX711_TIMEOUT");
    loadCellReady = false;
  } else {
    LoadCell.setCalFactor(LOADCELL_CAL_FACTOR);

    long savedOffset = prefs.getLong(PREFS_KEY_TARE, TARE_SENTINEL);
    if (savedOffset != TARE_SENTINEL) {
      // Đã có offset cũ → khôi phục, không tare. Vật đặt sẵn sẽ hiển thị đúng kg.
      LoadCell.setTareOffset(savedOffset);
      Serial.print("OK:TARE_RESTORED,");
      Serial.println(savedOffset);
    } else {
      // Lần đầu chạy (NVS rỗng) → tare cứng để có gốc 0, rồi lưu vào NVS.
      LoadCell.tare();  // blocking
      prefs.putLong(PREFS_KEY_TARE, LoadCell.getTareOffset());
      Serial.println("OK:TARE_INITIAL");
    }
    loadCellReady = true;
  }

  ESP32PWM::allocateTimer(0);
  ESP32PWM::allocateTimer(1);
  panServo.setPeriodHertz(50);
  tiltServo.setPeriodHertz(50);
  panServo.attach(PAN_SERVO_PIN, SERVO_MIN_US, SERVO_MAX_US);
  tiltServo.attach(TILT_SERVO_PIN, SERVO_MIN_US, SERVO_MAX_US);

  panServo.write(PAN_HOME);
  tiltServo.write(TILT_HOME);
  currentPan  = PAN_HOME;
  currentTilt = TILT_HOME;

  delay(500);
  Serial.println("READY");
}

// ============== MAIN LOOP ==============
void loop() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      if (inputBuffer.length() > 0) {
        processCommand(inputBuffer);
        inputBuffer = "";
      }
    } else {
      inputBuffer += c;
    }
  }

  // HX711_ADC: gọi update() càng nhanh càng tốt để bộ lọc trượt hoạt động.
  if (loadCellReady) {
    LoadCell.update();
    if (LoadCell.getTareStatus()) {
      // Tare từ host vừa hoàn tất → lưu offset mới vào NVS để lần boot sau dùng lại.
      prefs.putLong(PREFS_KEY_TARE, LoadCell.getTareOffset());
      Serial.println("OK:TARE_COMPLETE");
    }
  }

  unsigned long now = millis();
  if (now - lastUpdateTime >= UPDATE_INTERVAL) {
    lastUpdateTime = now;
    updateServos();
  }

  updatePumpState(now);

  if (now - lastSensorTime >= SENSOR_INTERVAL) {
    lastSensorTime = now;
    readAndSendSensors();
  }
}

// ============== PUMP STATE MACHINE ==============
// IDLE → (FIRE) → FIRING (5s, relay ON) → COOLDOWN (5s, relay OFF) → IDLE
void updatePumpState(unsigned long now) {
  switch (pumpState) {
    case PUMP_IDLE:
      // Chờ lệnh FIRE từ processCommand().
      break;
    case PUMP_FIRING:
      if (now - pumpStateEntered >= PUMP_FIRE_MS) {
        pumpRelayOff();
        pumpState = PUMP_COOLDOWN;
        pumpStateEntered = now;
        Serial.println("OK:PUMP_STOP");
      }
      break;
    case PUMP_COOLDOWN:
      if (now - pumpStateEntered >= PUMP_COOLDOWN_MS) {
        pumpState = PUMP_IDLE;
        Serial.println("OK:PUMP_READY");
      }
      break;
  }
}

// ============== COMMAND PROCESSING ==============
void processCommand(String cmd) {
  cmd.trim();

  if (cmd == "PING") {
    Serial.println("PONG");
    return;
  }

  if (cmd == "HOME") {
    targetPan  = PAN_HOME;
    targetTilt = TILT_HOME;
    Serial.println("OK:HOME");
    return;
  }

  if (cmd == "PUMP,FIRE") {
    if (pumpState == PUMP_IDLE) {
      pumpRelayOn();
      pumpState = PUMP_FIRING;
      pumpStateEntered = millis();
      Serial.println("OK:PUMP_FIRE");
    } else {
      // Đang FIRING hoặc COOLDOWN → host gửi loạn cũng không kích lại.
      Serial.println("BUSY:PUMP");
    }
    return;
  }

  if (cmd == "TARE") {
    if (loadCellReady) {
      LoadCell.tareNoDelay();
      Serial.println("OK:TARE");
    } else {
      Serial.println("ERR:HX711_NOT_READY");
    }
    return;
  }

  // Parse "PAN,TILT"
  int commaIndex = cmd.indexOf(',');
  if (commaIndex > 0) {
    float pan  = cmd.substring(0, commaIndex).toFloat();
    float tilt = cmd.substring(commaIndex + 1).toFloat();
    pan  = constrain(pan,  PAN_MIN,  PAN_MAX);
    tilt = constrain(tilt, TILT_MIN, TILT_MAX);
    targetPan  = pan;
    targetTilt = tilt;
    Serial.print("OK:");
    Serial.print(pan, 1);
    Serial.print(",");
    Serial.println(tilt, 1);
  }
}

// ============== SERVO SMOOTHING ==============
void updateServos() {
  currentPan  = currentPan  + SMOOTH_FACTOR * (targetPan  - currentPan);
  currentTilt = currentTilt + SMOOTH_FACTOR * (targetTilt - currentTilt);
  panServo.write((int)currentPan);
  tiltServo.write((int)currentTilt);
}

// ============== SENSOR READING ==============
float readDistanceCm() {
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);

  long duration = pulseIn(ECHO_PIN, HIGH, 30000);
  if (duration == 0) return -1.0f;

  float distance = (duration * 0.0343f) / 2.0f;
  if (distance < 2.0f || distance > 400.0f) return -1.0f;
  return distance;
}

float readWeightKg() {
  if (!loadCellReady) return -999.0f;
  // HX711_ADC luôn trả giá trị đã smooth bởi LoadCell.update() trong loop.
  // Đảo dấu để khi đè lên load cell hiển thị giá trị dương (khỏi phải đảo dây E+/E-).
  float w = -LoadCell.getData();
  // Deadband ±0.05 kg: triệt noise quanh zero để khỏi hiển thị "-0.00" / "-0.01"
  if (w > -0.05f && w < 0.05f) w = 0.0f;
  return w;
}

void readAndSendSensors() {
  // Adafruit DHT có cache 2s nội bộ — đọc 1 lần / chu kỳ là đủ.
  // Đọc humidity trước rồi temperature: cả hai dùng chung 1 lần đo bit-banged.
  float humidity    = dht.readHumidity();
  float temperature = dht.readTemperature();
  if (isnan(temperature)) temperature = -999.0f;
  if (isnan(humidity))    humidity    = -999.0f;

  float distance = readDistanceCm();
  float weight   = readWeightKg();

  Serial.print("SENSOR:");
  Serial.print(temperature, 1);
  Serial.print(",");
  Serial.print(humidity, 1);
  Serial.print(",");
  Serial.print(distance, 1);
  Serial.print(",");
  Serial.println(weight, 2);
}
