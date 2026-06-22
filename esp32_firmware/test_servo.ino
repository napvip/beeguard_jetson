/*
 * Test Servo - Quay qua quay lại quanh HOME
 * Pan (GPIO 26), Tilt (GPIO 25)
 */
#include <ESP32Servo.h>

#define PAN_PIN   26
#define TILT_PIN  25
#define HOME      90
#define SWEEP     30    // quay ±30° quanh home
#define STEP_MS   15    // delay mỗi bước (nhỏ = nhanh)

Servo pan, tilt;
int angle = HOME;
int dir = 1;

void setup() {
  Serial.begin(115200);
  ESP32PWM::allocateTimer(0);
  ESP32PWM::allocateTimer(1);
  pan.setPeriodHertz(50);
  tilt.setPeriodHertz(50);
  pan.attach(PAN_PIN, 500, 2500);
  tilt.attach(TILT_PIN, 500, 2500);
  pan.write(HOME);
  tilt.write(HOME);
  delay(1000);
  Serial.println("Test servo start");
}

void loop() {
  angle += dir;
  if (angle >= HOME + SWEEP || angle <= HOME - SWEEP) {
    dir = -dir;
  }
  pan.write(angle);
  tilt.write(angle);
  Serial.println(angle);
  delay(STEP_MS);
}
