/*
 * ============================================================
 *  ESP32 + PCA9685 Servo Controller
 *  Gesture-Controlled 6-DOF Robotic Manipulator
 * ============================================================
 *
 *  RECEIVES from p1.py over USB Serial (115200 baud):
 *    "base,shoulder,elbow,yaw,roll,jaws\n"
 *    Each value is a float angle in degrees (0.0 - 180.0)
 *
 *  HARDWARE:
 *    - ESP32 DevKit (classic)
 *    - PCA9685 16-channel PWM servo driver (I2C)
 *    - 6x MG90S micro servos
 *    - 5V 10A external PSU for servos (connected to PCA9685 V+)
 *
 *  WIRING (ESP32 -> PCA9685):
 *    GPIO 21 (SDA) -> SDA
 *    GPIO 22 (SCL) -> SCL
 *    GND           -> GND
 *    (Do NOT power servos from ESP32 — use external 5V PSU on PCA9685 V+ terminal)
 *
 *  PCA9685 CHANNEL MAPPING:
 *    CH 0 -> Base rotation
 *    CH 1 -> Shoulder
 *    CH 2 -> Elbow
 *    CH 3 -> Yaw (end-effector)
 *    CH 4 -> Roll (end-effector)
 *    CH 5 -> Jaws (gripper)
 *
 *  LIBRARY REQUIRED:
 *    Install "Adafruit PWM Servo Driver Library" from Arduino Library Manager
 *    (It will also install "Adafruit BusIO" as a dependency)
 *
 *  BOARD SETUP in Arduino IDE:
 *    1. File -> Preferences -> Additional Board Manager URLs:
 *       https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json
 *    2. Tools -> Board -> Board Manager -> search "esp32" -> install
 *    3. Tools -> Board -> ESP32 Arduino -> "ESP32 Dev Module"
 *    4. Tools -> Upload Speed -> 921600
 *    5. Tools -> Port -> (your ESP32 USB port)
 */

#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

// ================================================================
//  PCA9685 SETUP
// ================================================================
// Default I2C address is 0x40. Change if you've soldered address jumpers.
Adafruit_PWMServoDriver pca = Adafruit_PWMServoDriver(0x40);

#define PWM_FREQ 50   // 50 Hz for standard servos (20ms period)

// ================================================================
//  MG90S SERVO PULSE WIDTH CALIBRATION
//  Adjust these if your servos don't reach full 0-180 range
// ================================================================
// PCA9685 has 4096 ticks per 20ms cycle at 50Hz
// MG90S typical: 500µs (0°) to 2400µs (180°)
//   500  / 20000 * 4096 ≈ 102
//   2400 / 20000 * 4096 ≈ 491

#define SERVO_MIN_TICK  102    // ~500µs  -> 0 degrees
#define SERVO_MAX_TICK  491    // ~2400µs -> 180 degrees

// ================================================================
//  CHANNEL MAPPING  (edit if your wiring order differs)
// ================================================================
#define CH_BASE      0
#define CH_SHOULDER  1
#define CH_ELBOW     2
#define CH_YAW       3
#define CH_ROLL      4
#define CH_JAWS      5

#define NUM_SERVOS   6

const int SERVO_CHANNELS[NUM_SERVOS] = {
  CH_BASE, CH_SHOULDER, CH_ELBOW, CH_YAW, CH_ROLL, CH_JAWS
};

// Joint names for debug printing
const char* JOINT_NAMES[NUM_SERVOS] = {
  "BASE", "SHLDR", "ELBOW", "YAW", "ROLL", "JAWS"
};

// ================================================================
//  SMOOTHING / INTERPOLATION
// ================================================================
// Smoothing factor (0.0 = no movement, 1.0 = instant snap)
// Lower = smoother but slower response, Higher = faster but jerkier
#define SMOOTH_FACTOR 0.3

float currentAngles[NUM_SERVOS] = {90.0, 90.0, 90.0, 90.0, 90.0, 40.0};
float targetAngles[NUM_SERVOS]  = {90.0, 90.0, 90.0, 90.0, 90.0, 40.0};

// ================================================================
//  SERIAL PARSING
// ================================================================
#define SERIAL_BAUD    115200
#define MAX_LINE_LEN   64

char lineBuffer[MAX_LINE_LEN];
int  lineIndex = 0;

// ================================================================
//  HELPER FUNCTIONS
// ================================================================

/**
 * Convert angle (0-180 degrees) to PCA9685 tick value
 */
uint16_t angleToPWM(float angle) {
  angle = constrain(angle, 0.0, 180.0);
  return (uint16_t)map((long)(angle * 10), 0, 1800, SERVO_MIN_TICK, SERVO_MAX_TICK);
}

/**
 * Set a single servo to a specific angle
 */
void setServo(int channel, float angle) {
  uint16_t tick = angleToPWM(angle);
  pca.setPWM(channel, 0, tick);
}

/**
 * Parse incoming serial line: "base,shoulder,elbow,yaw,roll,jaws"
 * Returns true if 6 values were successfully parsed
 */
bool parseLine(const char* line, float* angles) {
  char buf[MAX_LINE_LEN];
  strncpy(buf, line, MAX_LINE_LEN - 1);
  buf[MAX_LINE_LEN - 1] = '\0';

  int count = 0;
  char* token = strtok(buf, ",");
  while (token != NULL && count < NUM_SERVOS) {
    angles[count] = atof(token);
    // Sanity check
    if (angles[count] < 0.0 || angles[count] > 180.0) {
      angles[count] = constrain(angles[count], 0.0, 180.0);
    }
    count++;
    token = strtok(NULL, ",");
  }
  return (count == NUM_SERVOS);
}

/**
 * Startup sweep test — moves each servo slowly to verify wiring
 */
void startupTest() {
  Serial.println("[INFO] Running startup servo test...");
  
  // Move all to 90 (rest)
  for (int i = 0; i < NUM_SERVOS; i++) {
    setServo(SERVO_CHANNELS[i], 90.0);
  }
  delay(500);

  // Sweep each servo: 90 -> 60 -> 120 -> 90
  for (int i = 0; i < NUM_SERVOS; i++) {
    Serial.print("[TEST] ");
    Serial.println(JOINT_NAMES[i]);
    
    for (float a = 90; a >= 60; a -= 2) {
      setServo(SERVO_CHANNELS[i], a);
      delay(20);
    }
    for (float a = 60; a <= 120; a += 2) {
      setServo(SERVO_CHANNELS[i], a);
      delay(20);
    }
    for (float a = 120; a >= 90; a -= 2) {
      setServo(SERVO_CHANNELS[i], a);
      delay(20);
    }
    delay(100);
  }
  Serial.println("[INFO] Startup test complete.");
}

// ================================================================
//  SETUP
// ================================================================
void setup() {
  Serial.begin(SERIAL_BAUD);
  delay(100);
  Serial.println();
  Serial.println("================================");
  Serial.println(" ESP32 + PCA9685 Servo Controller");
  Serial.println(" Gesture Control Receiver");
  Serial.println("================================");

  // Initialize I2C and PCA9685
  Wire.begin(21, 22);        // SDA=21, SCL=22 (ESP32 default)
  Wire.setClock(400000);     // 400kHz I2C fast mode for snappy response
  
  pca.begin();
  pca.setOscillatorFrequency(25000000);  // Internal oscillator (calibrated)
  pca.setPWMFreq(PWM_FREQ);
  delay(10);

  Serial.println("[INFO] PCA9685 initialized at 50Hz");

  // Set initial resting positions
  float restAngles[NUM_SERVOS] = {90.0, 90.0, 90.0, 90.0, 90.0, 40.0};
  for (int i = 0; i < NUM_SERVOS; i++) {
    currentAngles[i] = restAngles[i];
    targetAngles[i]  = restAngles[i];
    setServo(SERVO_CHANNELS[i], restAngles[i]);
  }
  Serial.println("[INFO] Servos set to resting position");

  // Optional: run startup sweep test (comment out to skip)
  // startupTest();

  Serial.println("[INFO] Waiting for gesture commands...");
  Serial.println("[INFO] Format: base,shoulder,elbow,yaw,roll,jaws");
  Serial.println("[READY]");  // Handshake signal for p1.py
}

// ================================================================
//  MAIN LOOP
// ================================================================
void loop() {
  // --- 1. Read serial data (non-blocking) ---
  while (Serial.available() > 0) {
    char c = Serial.read();
    
    if (c == '\n' || c == '\r') {
      if (lineIndex > 0) {
        lineBuffer[lineIndex] = '\0';
        
        // Parse the received line
        float newAngles[NUM_SERVOS];
        if (parseLine(lineBuffer, newAngles)) {
          // Update target angles
          for (int i = 0; i < NUM_SERVOS; i++) {
            targetAngles[i] = newAngles[i];
          }
          // Debug: confirm receipt (comment out for production to reduce latency)
          Serial.print("[OK] ");
          for (int i = 0; i < NUM_SERVOS; i++) {
            Serial.print(newAngles[i], 1);
            if (i < NUM_SERVOS - 1) Serial.print(",");
          }
          Serial.println();
        } else {
          Serial.print("[ERR] Bad parse: ");
          Serial.println(lineBuffer);
        }
        lineIndex = 0;
      }
    } else {
      if (lineIndex < MAX_LINE_LEN - 1) {
        lineBuffer[lineIndex++] = c;
      } else {
        // Buffer overflow — reset
        lineIndex = 0;
      }
    }
  }

  // --- 2. Smoothly interpolate current -> target ---
  bool moved = false;
  for (int i = 0; i < NUM_SERVOS; i++) {
    float diff = targetAngles[i] - currentAngles[i];
    if (abs(diff) > 0.3) {   // dead-zone to avoid jitter
      currentAngles[i] += diff * SMOOTH_FACTOR;
      currentAngles[i] = constrain(currentAngles[i], 0.0, 180.0);
      setServo(SERVO_CHANNELS[i], currentAngles[i]);
      moved = true;
    }
  }

  // --- 3. Small delay to control loop rate ---
  // ~50Hz update rate (matches p1.py serial send rate)
  delay(20);
}
